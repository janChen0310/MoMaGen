# MoMaGen + TidyBot++ data-generation environment (Isaac Sim 4.5 / OmniGibson 3.7.1)
#
# The image contains the full Python environment (Isaac Sim, torch, CuRobo
# compiled for Ampere+); the MoMaGen monorepo and the BEHAVIOR-1K datasets are
# MOUNTED at runtime and editable-installed on first container start (see
# entrypoint.sh). Build once on the server:
#
#   cd MoMaGen && docker build -t momagen-tidybot -f docker/Dockerfile docker/
#
# Requires the NVIDIA Container Toolkit and a non-Blackwell GPU (e.g. RTX A6000;
# Isaac Sim 4.5 deadlocks on RTX 50-series / sm_120).

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    OMNI_KIT_ACCEPT_EULA=YES \
    ACCEPT_EULA=Y \
    OMNIGIBSON_HEADLESS=1 \
    # Ampere (A6000) + Ada + Hopper; extend if other GPUs are used
    TORCH_CUDA_ARCH_LIST="8.6;8.9;9.0"

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        git git-lfs wget curl ca-certificates \
        # Omniverse Kit runtime libraries
        libvulkan1 vulkan-tools libglu1-mesa libegl1 libgl1 libxt6 libxrandr2 \
        libxinerama1 libxcursor1 libxi6 libsm6 libice6 libfontconfig1 \
        ninja-build cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python3.10 -m pip install --no-cache-dir --upgrade pip setuptools wheel

# Torch first (CuRobo compiles against it). cu128 builds cover sm_86/89/90.
RUN python3.10 -m pip install --no-cache-dir torch==2.11.* \
        --index-url https://download.pytorch.org/whl/cu128

# Isaac Sim 4.5 via NVIDIA's pip index (same version the momagen conda env pins)
RUN python3.10 -m pip install --no-cache-dir "isaacsim[all,extscache]==4.5.0.0" \
        --extra-index-url https://pypi.nvidia.com

# Pinned third-party deps from the working momagen env (docker/momagen-pip-freeze.lock,
# with editable/local-wheel lines stripped). Keep numpy<2 per OmniGibson.
COPY momagen-pip-freeze.lock /tmp/momagen-pip-freeze.lock
RUN grep -vE "file:///|^(torch|nvidia-curobo|ompl)" /tmp/momagen-pip-freeze.lock > /tmp/reqs.txt \
    && python3.10 -m pip install --no-cache-dir -r /tmp/reqs.txt || \
       echo "WARNING: some pinned deps failed; editable installs will pull their own"

# CuRobo (StanfordVL fork pinned by OmniGibson's [primitives] extra) + OMPL
RUN python3.10 -m pip install --no-cache-dir \
        "nvidia-curobo @ git+https://github.com/StanfordVL/curobo@d083916436f1f749cdd351bdf96c4c5df4e6d192" \
        "ompl @ https://storage.googleapis.com/gibson_scenes/ompl-1.6.0-cp310-cp310-manylinux_2_28_x86_64.whl"

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /workspace/MoMaGen
ENTRYPOINT ["/entrypoint.sh"]
CMD ["bash"]
