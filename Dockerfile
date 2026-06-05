# Tier R heavy image. One recipe, two worlds via BASE_IMAGE.
#   CPU  : docker build -t forge-exp:cpu .
#   CUDA : docker build --build-arg BASE_IMAGE=nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04 -t forge-exp:cuda .
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

# The audio trap: librosa silently fails to decode GTZAN without these.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git && rm -rf /var/lib/apt/lists/*

# uv for fast, locked installs
RUN pip install --no-cache-dir uv

WORKDIR /workspace
# Lockfile FIRST -> uv sync layer survives source edits (fast rebuilds).
COPY pyproject.toml uv.lock* ./
RUN uv sync --frozen || uv sync     # falls back if no lock yet (first build)

COPY . .
ENV PYTHONPATH=/workspace
CMD ["bash"]
