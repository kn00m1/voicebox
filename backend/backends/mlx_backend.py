"""
MLX backend implementation for TTS and STT using mlx-audio.
"""

from typing import Optional, List, Tuple
import asyncio
import logging
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)

# PATCH: Import and apply offline patch BEFORE any huggingface_hub usage
# This prevents mlx_audio from making network requests when models are cached
from ..utils.hf_offline_patch import patch_huggingface_hub_offline, ensure_original_qwen_config_cached

patch_huggingface_hub_offline()
ensure_original_qwen_config_cached()

from . import TTSBackend, STTBackend, LANGUAGE_CODE_TO_NAME, WHISPER_HF_REPOS
from .base import is_model_cached, combine_voice_prompts as _combine_voice_prompts, model_load_progress
from ..utils.cache import get_cache_key, get_cached_voice_prompt, cache_voice_prompt
from ..utils.hf_offline_patch import force_offline_if_cached


_tts_semaphore: Optional[asyncio.Semaphore] = None


def _get_tts_semaphore() -> asyncio.Semaphore:
    """Return a process-wide semaphore that serializes MPS inference."""
    global _tts_semaphore
    if _tts_semaphore is None:
        _tts_semaphore = asyncio.Semaphore(1)
    return _tts_semaphore


class MLXTTSBackend:
    """MLX-based TTS backend using mlx-audio."""

    def __init__(self, model_size: str = "1.7B"):
        self.model = None
        self.model_size = model_size
        self._current_model_size = None

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _get_model_path(self, model_size: str) -> str:
        """
        Get the MLX model path.

        Args:
            model_size: Model size (1.7B or 0.6B)

        Returns:
            HuggingFace Hub model ID for MLX
        """
        mlx_model_map = {
            "1.7B": "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16",
            "0.6B": "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
        }

        if model_size not in mlx_model_map:
            raise ValueError(f"Unknown model size: {model_size}")

        hf_model_id = mlx_model_map[model_size]
        logger.info("Will download MLX model from HuggingFace Hub: %s", hf_model_id)

        return hf_model_id

    def _is_model_cached(self, model_size: str) -> bool:
        return is_model_cached(
            self._get_model_path(model_size),
            weight_extensions=(".safetensors", ".bin", ".npz"),
        )

    async def load_model_async(self, model_size: Optional[str] = None):
        """
        Lazy load the MLX TTS model.

        Args:
            model_size: Model size to load (1.7B or 0.6B)
        """
        if model_size is None:
            model_size = self.model_size

        # If already loaded with correct size, return
        if self.model is not None and self._current_model_size == model_size:
            return

        # Unload existing model if different size requested
        if self.model is not None and self._current_model_size != model_size:
            self.unload_model()

        # Run blocking load in thread pool
        await asyncio.to_thread(self._load_model_sync, model_size)

    # Alias for compatibility
    load_model = load_model_async


    def _load_model_sync(self, model_size: str):
        """Synchronous model loading via qwen_tts (PyTorch)."""
        import torch
        from safetensors.torch import load_file
        from huggingface_hub import snapshot_download
        from qwen_tts import Qwen3TTSModel
        from qwen_tts.core.models import Qwen3TTSConfig, Qwen3TTSForConditionalGeneration, Qwen3TTSProcessor
        from transformers import AutoConfig, AutoProcessor

        model_path = self._get_model_path(model_size)
        model_name = f"qwen-tts-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(model_name, is_cached):
            logger.info("Loading Qwen3 TTS model %s via qwen_tts (PyTorch)...", model_size)
            local_path = snapshot_download(model_path, local_files_only=True)

            AutoConfig.register("qwen3_tts", Qwen3TTSConfig, exist_ok=True)
            AutoProcessor.register(Qwen3TTSConfig, Qwen3TTSProcessor, exist_ok=True)

            # Step 1: Load model; MLX conv weights have mismatched shapes so skip them.
            pt_model = Qwen3TTSForConditionalGeneration.from_pretrained(
                local_path,
                dtype=torch.bfloat16,
                local_files_only=True,
                ignore_mismatched_sizes=True,
            )

            # Step 2: Fix all 3D weights (MLX stores Conv1d as [out, kernel, in]; PyTorch expects [out, in, kernel]).
            raw_weights = load_file(f"{local_path}/model.safetensors")
            conv_fixes = {
                k: v.permute(0, 2, 1).contiguous().to(torch.bfloat16)
                for k, v in raw_weights.items()
                if v.ndim == 3
            }
            pt_model.load_state_dict(conv_fixes, strict=False)
            logger.info("Applied %d 3D weight transpositions for MLX→PyTorch", len(conv_fixes))

            processor = AutoProcessor.from_pretrained(local_path, fix_mistral_regex=True)
            generate_defaults = pt_model.generate_config

            device = "mps" if torch.backends.mps.is_available() else "cpu"
            pt_model = pt_model.to(device)

            wrapper = Qwen3TTSModel(model=pt_model, processor=processor, generate_defaults=generate_defaults)
            wrapper.device = torch.device(device)
            self.model = wrapper

        self._current_model_size = model_size
        self.model_size = model_size
        logger.info("Qwen3 TTS model %s loaded on %s", model_size, device)

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            self._current_model_size = None
            logger.info("MLX TTS model unloaded")

    async def create_voice_prompt(
        self,
        audio_path: str,
        reference_text: str,
        use_cache: bool = True,
    ) -> Tuple[dict, bool]:
        """
        Create voice prompt from reference audio.

        MLX backend stores voice prompt as a dict with audio path and text.
        The actual voice prompt processing happens during generation.

        Args:
            audio_path: Path to reference audio file
            reference_text: Transcript of reference audio
            use_cache: Whether to use cached prompt if available

        Returns:
            Tuple of (voice_prompt_dict, was_cached)
        """
        await self.load_model_async(None)

        # Check cache if enabled
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cached_prompt = get_cached_voice_prompt(cache_key)
            if cached_prompt is not None:
                # Return cached prompt (should be dict format)
                if isinstance(cached_prompt, dict):
                    # Validate that the cached audio file still exists
                    cached_audio_path = cached_prompt.get("ref_audio") or cached_prompt.get("ref_audio_path")
                    if cached_audio_path and Path(cached_audio_path).exists():
                        return cached_prompt, True
                    else:
                        # Cached file no longer exists, invalidate cache
                        logger.warning("Cached audio file not found: %s, regenerating prompt", cached_audio_path)

        # MLX voice prompt format - store audio path and text
        # The model will process this during generation
        voice_prompt_items = {
            "ref_audio": str(audio_path),
            "ref_text": reference_text,
        }

        # Cache if enabled
        if use_cache:
            cache_key = get_cache_key(audio_path, reference_text)
            cache_voice_prompt(cache_key, voice_prompt_items)

        return voice_prompt_items, False

    async def combine_voice_prompts(self, audio_paths, reference_texts):
        return await _combine_voice_prompts(audio_paths, reference_texts)

    async def generate(
        self,
        text: str,
        voice_prompt: dict,
        language: str = "en",
        seed: Optional[int] = None,
        instruct: Optional[str] = None,
    ) -> Tuple[np.ndarray, int]:
        """
        Generate audio from text using voice prompt.

        Args:
            text: Text to synthesize
            voice_prompt: Voice prompt dictionary with ref_audio and ref_text
            language: Language code (en or zh) - may not be fully supported by MLX
            seed: Random seed for reproducibility
            instruct: Natural language instruction (may not be supported by MLX)

        Returns:
            Tuple of (audio_array, sample_rate)
        """
        await self.load_model_async(None)

        logger.info("Generating audio for text: %s", text)

        def _generate_sync():
            """Run synchronous generation in thread pool."""
            import torch

            ref_audio = voice_prompt.get("ref_audio") or voice_prompt.get("ref_audio_path")
            ref_text = voice_prompt.get("ref_text", "")

            if ref_audio and not Path(ref_audio).exists():
                logger.warning("Audio file not found: %s — re-record voice profile", ref_audio)
                ref_audio = None

            if ref_audio is None:
                raise ValueError(
                    "Voice prompt reference audio not found. "
                    "Please re-record your voice profile."
                )

            lang_name = LANGUAGE_CODE_TO_NAME.get(language, "Auto")

            if seed is not None:
                import random
                torch.manual_seed(seed)
                random.seed(seed)
                np.random.seed(seed)

            # ICL mode uses ref_text for better voice cloning; x-vector-only when no transcript
            use_icl = bool(ref_text)

            wavs, sample_rate = self.model.generate_voice_clone(
                text=text,
                language=lang_name,
                ref_audio=ref_audio,
                ref_text=ref_text if use_icl else None,
                x_vector_only_mode=not use_icl,
                non_streaming_mode=True,
            )

            audio = wavs[0].astype(np.float32) if wavs else np.array([], dtype=np.float32)
            return audio, sample_rate

        # Serialize MPS inference — concurrent threads cause crashes on Apple Silicon
        async with _get_tts_semaphore():
            audio, sample_rate = await asyncio.to_thread(_generate_sync)

        return audio, sample_rate


class MLXSTTBackend:
    """MLX-based STT backend using mlx-audio Whisper."""

    def __init__(self, model_size: str = "base"):
        self.model = None
        self.model_size = model_size

    def is_loaded(self) -> bool:
        """Check if model is loaded."""
        return self.model is not None

    def _is_model_cached(self, model_size: str) -> bool:
        hf_repo = WHISPER_HF_REPOS.get(model_size, f"openai/whisper-{model_size}")
        return is_model_cached(hf_repo, weight_extensions=(".safetensors", ".bin", ".npz"))

    async def load_model_async(self, model_size: Optional[str] = None):
        """
        Lazy load the MLX Whisper model.

        Args:
            model_size: Model size (tiny, base, small, medium, large)
        """
        if model_size is None:
            model_size = self.model_size

        if self.model is not None and self.model_size == model_size:
            return

        # Run blocking load in thread pool
        await asyncio.to_thread(self._load_model_sync, model_size)

    # Alias for compatibility
    load_model = load_model_async

    def _load_model_sync(self, model_size: str):
        """Synchronous model loading."""
        progress_model_name = f"whisper-{model_size}"
        is_cached = self._is_model_cached(model_size)

        with model_load_progress(progress_model_name, is_cached):
            from mlx_audio.stt import load

            model_name = WHISPER_HF_REPOS.get(model_size, f"openai/whisper-{model_size}")
            logger.info("Loading MLX Whisper model %s...", model_size)

            with force_offline_if_cached(is_cached, progress_model_name):
                self.model = load(model_name)

        self.model_size = model_size
        logger.info("MLX Whisper model %s loaded successfully", model_size)

    def unload_model(self):
        """Unload the model to free memory."""
        if self.model is not None:
            del self.model
            self.model = None
            logger.info("MLX Whisper model unloaded")

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
    ) -> str:
        """
        Transcribe audio to text.

        Args:
            audio_path: Path to audio file
            language: Optional language hint
            model_size: Optional model size override

        Returns:
            Transcribed text
        """
        await self.load_model_async(model_size)

        progress_model_name = f"whisper-{self.model_size}"

        def _transcribe_sync():
            """Run synchronous transcription in thread pool."""
            # MLX Whisper transcription using generate method
            # The generate method accepts audio path directly
            decode_options = {}
            if language:
                decode_options["language"] = language

            # Model is loaded → weights are on disk. Force offline so
            # lazy tokenizer/config lookups don't hang when the user is
            # disconnected (issue #462).
            with force_offline_if_cached(True, progress_model_name):
                result = self.model.generate(str(audio_path), **decode_options)

            # Extract text from result
            if isinstance(result, str):
                return result.strip()
            elif isinstance(result, dict):
                return result.get("text", "").strip()
            elif hasattr(result, "text"):
                return result.text.strip()
            else:
                return str(result).strip()

        # Run blocking transcription in thread pool
        return await asyncio.to_thread(_transcribe_sync)
