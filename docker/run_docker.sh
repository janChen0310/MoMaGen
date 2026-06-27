#!/usr/bin/env bash
# Launch the momagen-tidybot container on the GPU server.
#
#   ./docker/run_docker.sh                          # interactive shell
#   ./docker/run_docker.sh python momagen/scripts/generate_dataset.py ...
#
# Expects this script to live in <MoMaGen-checkout>/docker/. Mounts:
#   - the MoMaGen checkout       -> /workspace/MoMaGen
#   - tidybot_platform (sibling) -> /workspace/tidybot_platform
#   - BEHAVIOR-1K datasets stay inside the checkout (BEHAVIOR-1K/datasets)
# X11 is forwarded when DISPLAY is set (needed for the teleop GUI; datagen is headless).
set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIDYBOT_DIR="$(dirname "$REPO_DIR")/tidybot_platform"
IMAGE=momagen-tidybot

X11_ARGS=()
if [ -n "$DISPLAY" ]; then
    X11_ARGS=(-e DISPLAY="$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix:rw)
    xhost +local:docker >/dev/null 2>&1 || true
fi

docker run --rm -it \
    --gpus all \
    --network host \
    --shm-size 16g \
    -e OMNI_KIT_ACCEPT_EULA=YES \
    -e OMNIGIBSON_HEADLESS="${OMNIGIBSON_HEADLESS:-1}" \
    -v "$REPO_DIR":/workspace/MoMaGen \
    -v "$TIDYBOT_DIR":/workspace/tidybot_platform \
    "${X11_ARGS[@]}" \
    "$IMAGE" "$@"
