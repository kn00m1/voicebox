"""
STT backend that delegates to a local whisper.cpp binary with GGML model files.

Avoids re-downloading models already available in GGML format.
Implements the STTBackend protocol from backends/base.py.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# Default paths — mirror what Thinking Out Loud uses
DEFAULT_WHISPER_BIN = Path.home() / "whisper.cpp" / "build" / "bin" / "whisper-cli"
DEFAULT_MODELS_DIR = Path.home() / "whisper.cpp" / "models"

# Map Voicebox model_size keys → GGML filenames
GGML_MODEL_MAP: dict[str, str] = {
    "turbo": "ggml-large-v3-turbo.bin",
    "large": "ggml-large-v3.bin",
    "medium": "ggml-medium.bin",
    "small": "ggml-small.bin",
    "base": "ggml-base.bin",
    "tiny": "ggml-tiny.bin",
}


def _find_whisper_bin() -> Optional[Path]:
    path = Path(os.environ.get("WHISPER_CPP_BIN", DEFAULT_WHISPER_BIN))
    return path if path.exists() else None


def _find_models_dir() -> Path:
    return Path(os.environ.get("WHISPER_CPP_MODELS_DIR", DEFAULT_MODELS_DIR))


class WhisperCppSTTBackend:
    """
    STT backend using the local whisper-cli binary and GGML model files.

    No HuggingFace download — reads from ~/whisper.cpp/models/*.bin.
    Accepted as a drop-in wherever STTBackend is used because it
    satisfies the structural Protocol (load_model, transcribe,
    unload_model, is_loaded, _is_model_cached).
    """

    def __init__(self):
        self._loaded = False
        self.model_size: Optional[str] = None
        self._bin = _find_whisper_bin()
        self._models_dir = _find_models_dir()

    # ── Protocol helpers ──────────────────────────────────────────────

    def is_loaded(self) -> bool:
        return self._loaded

    def _is_model_cached(self, model_size: str) -> bool:
        """Return True if the GGML file for this size exists on disk."""
        filename = GGML_MODEL_MAP.get(model_size)
        if not filename:
            return False
        return (self._models_dir / filename).exists()

    # ── Load / unload ─────────────────────────────────────────────────

    async def load_model(self, model_size: Optional[str] = None) -> None:
        """
        'Loading' for whisper-cli is instant — just validate the binary
        and model file exist; no weights are kept resident between calls.
        """
        size = model_size or "turbo"
        if not self._bin:
            raise FileNotFoundError(
                f"whisper-cli binary not found at {DEFAULT_WHISPER_BIN}. "
                "Build whisper.cpp or set WHISPER_CPP_BIN env var."
            )
        if not self._is_model_cached(size):
            filename = GGML_MODEL_MAP.get(size, f"ggml-{size}.bin")
            raise FileNotFoundError(
                f"GGML model not found: {self._models_dir / filename}. "
                "Run: cd ~/whisper.cpp && bash models/download-ggml-model.sh large-v3-turbo"
            )
        self.model_size = size
        self._loaded = True
        logger.info("WhisperCppSTTBackend ready: %s (%s)", self._bin, GGML_MODEL_MAP[size])

    # Alias so both sync-style callers and async callers work
    load_model_async = load_model

    def unload_model(self) -> None:
        self._loaded = False
        self.model_size = None

    # ── Transcription ─────────────────────────────────────────────────

    async def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = None,
        model_size: Optional[str] = None,
    ) -> str:
        """
        Run whisper-cli on audio_path and return the transcript.

        whisper-cli is stateless — each call spawns a fresh process.
        This matches what Thinking Out Loud does for final transcription.
        """
        await self.load_model(model_size or self.model_size or "turbo")

        size = model_size or self.model_size or "turbo"
        ggml_path = str(self._models_dir / GGML_MODEL_MAP[size])

        cmd = [
            str(self._bin),
            "-m", ggml_path,
            "-f", str(audio_path),
            "-nt",          # no timestamps
            "--no-prints",  # suppress progress output
        ]
        if language:
            cmd += ["-l", language]

        logger.debug("whisper-cli cmd: %s", " ".join(cmd))

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()

        if proc.returncode != 0:
            err = stderr.decode(errors="replace").strip()
            raise RuntimeError(f"whisper-cli exited {proc.returncode}: {err}")

        return stdout.decode(errors="replace").strip()
