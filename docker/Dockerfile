FROM nvidia/cuda:11.8.0-base-ubuntu20.04

ARG DEBIAN_FRONTEND=noninteractive

# Install base utilities
RUN apt-get update \
    && apt-get install -y build-essential wget git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
ENV PATH=$CONDA_DIR/bin:$PATH
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p $CONDA_DIR

# Create conda environment and install Python
RUN conda create -n mmaction2 python=3.8 -y
ENV PATH=$CONDA_DIR/envs/mmaction2/bin:$PATH

# Ensure the environment is activated by using its binaries directly
RUN python -m pip install --upgrade pip

# Set environment variables
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 ffmpeg \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
# Install PyTorch, MMCV, and dependencies
RUN pip install torch==2.2.2+cu118 torchvision==0.17.2+cu118 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu118
RUN pip install openmim
RUN mim install mmengine mmcv

# Install MMAction2
WORKDIR /mmaction2
RUN git clone https://github.com/open-mmlab/mmaction2.git . && \
    mkdir -p ./data && \
    git checkout main && \
    pip install cython --no-cache-dir && \
    pip install --no-cache-dir -e .

# Adjust the work directory if you need to run commands directly within the environment
WORKDIR /mmaction2
