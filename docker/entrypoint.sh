#!/usr/bin/env bash
# First-run setup inside the momagen-tidybot container: editable-install the
# mounted repos (deps are already baked into the image), then exec the command.
set -e

MARKER=/workspace/.momagen_setup_done

if [ ! -f "$MARKER" ]; then
    echo "[entrypoint] First run: editable-installing mounted repos..."
    for pkg in \
        /workspace/MoMaGen/BEHAVIOR-1K/bddl \
        /workspace/MoMaGen/BEHAVIOR-1K/OmniGibson \
        /workspace/MoMaGen/robomimic \
        /workspace/MoMaGen; do
        if [ -d "$pkg" ]; then
            python3.10 -m pip install --no-deps --no-build-isolation -e "$pkg"
        else
            echo "[entrypoint] WARNING: $pkg not mounted; skipping"
        fi
    done
    touch "$MARKER"
    echo "[entrypoint] Setup done."
fi

exec "$@"
