FROM nvidia/cuda:11.4.3-devel-ubuntu20.04

RUN apt-get update && apt-get install wget -yq
RUN apt-get install build-essential g++ gcc -y
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get install libgl1-mesa-glx libglib2.0-0 -y
RUN apt-get install openmpi-bin openmpi-common libopenmpi-dev libgtk2.0-dev git -y

# Install miniconda with Python 3.8 pre-installed (avoids TOS and version conflicts)
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-py38_4.12.0-Linux-x86_64.sh -O ~/miniconda.sh && \
     /bin/bash ~/miniconda.sh -b -p /opt/conda
# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH
# Auto-accept Anaconda TOS in Docker (official solution)
ENV CONDA_PLUGINS_AUTO_ACCEPT_TOS=true
# Python 3.8 is already included, no need to install separately
# Use pip for PyTorch installation
RUN pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 --extra-index-url https://download.pytorch.org/whl/cu113
# Note: PyTorch 1.10.2 doesn't have official cu114 wheels, but cu113 wheels work with CUDA 11.4
# For spconv, use cu114 version
RUN pip install spconv-cu114
RUN pip install Pillow==8.4.0
RUN pip install tqdm
RUN pip install torchpack
# Install mmcv-full with pre-built wheels (avoids building from source)
# Limit parallel compilation jobs to prevent RAM/CPU exhaustion
ENV MAX_JOBS=1
# Use pre-built wheel from OpenMMLab (for CUDA 11.4 + PyTorch 1.10.x)
# Note: cu113 wheels are compatible with CUDA 11.4, but trying cu114 first
# This avoids building from source which consumes massive resources
RUN pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu114/torch1.10.0/index.html || \
    pip install mmcv-full==1.4.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html
# Install mmcv and mmdet separately
RUN pip install mmcv==1.4.0 mmdet==2.20.0
RUN pip install nuscenes-devkit
RUN pip install mpi4py==3.0.3
# Upgrade numba to version compatible with newer numpy (numba 0.56+ supports numpy 1.22+)
# numba 0.48.0 is too old and incompatible with numpy 1.20+
RUN pip install "numba>=0.56.0"
