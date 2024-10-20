#!/bin/bash

PROJ_ROOT=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Create conda environment
conda create -n foundationpose_ros python=3.10 -y

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate foundationpose_ros

# Install dependencies
pip install torchvision==0.16.0+cu121 torchaudio==2.1.0 torch==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
python -m pip install -r requirements.txt

# Clone source repository of FoundationPose
git clone https://github.com/NVlabs/FoundationPose.git

# Create the weights directory if it doesn't exist
mkdir -p weights

# Download pretrained weights
gdown --folder https://drive.google.com/drive/folders/1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i -O weights

# Install pybind11
cd ${PROJ_ROOT}/FoundationPose && git clone https://github.com/pybind/pybind11 && \
    cd pybind11 && git checkout v2.10.0 && \
    mkdir build && cd build && cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF && \
    make -j6 && make install

# Install Eigen
cd ${PROJ_ROOT}/FoundationPose && wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz && \
    tar xvzf ./eigen-3.4.0.tar.gz && \
    cd eigen-3.4.0 && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make install

# Clone and install nvdiffrast
cd ${PROJ_ROOT}/FoundationPose && git clone https://github.com/NVlabs/nvdiffrast && \
    conda activate foundationpose_ros && cd /nvdiffrast && pip install .

# Install mycpp
cd ${PROJ_ROOT}/FoundationPose/mycpp/ && \
rm -rf build && mkdir -p build && cd build && \
cmake .. && \
make -j$(nproc)

# Install mycuda
cd ${PROJ_ROOT}/FoundationPose/bundlesdf/mycuda && \
rm -rf build *egg* *.so && \
python3 -m pip install -e .

cd ${PROJ_ROOT}
