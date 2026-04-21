#!/bin/bash
# Replacement for /Applications/Voicebox.app/Contents/MacOS/voicebox-server
# Runs the modified Python backend (with whisper-cpp STT support) instead of
# the compiled PyInstaller binary.

FORK_DIR="$HOME/Documents/PARA/Projects/voicebox"
VENV="$FORK_DIR/backend/.venv/bin/python"

# Strip --parent-pid <pid> from args (not supported by Python argparser)
ARGS=()
skip_next=false
for arg in "$@"; do
    if $skip_next; then
        skip_next=false
        continue
    fi
    if [[ "$arg" == "--parent-pid" ]]; then
        skip_next=true
        continue
    fi
    ARGS+=("$arg")
done

cd "$FORK_DIR"
exec "$VENV" -m backend.main "${ARGS[@]}"
