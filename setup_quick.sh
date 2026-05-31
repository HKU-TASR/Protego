#!/bin/bash
# Quick environment + asset setup for Protego.
# Creates a conda env named "protego", installs all dependencies (including SMIRK
# and PyTorch3D), downloads the third-party detector weights, and then fetches the
# large datasets/checkpoints via tools/download_assets.py.
set -e

OS_TYPE=$(uname)
if [[ "$OS_TYPE" == "Darwin" ]]; then
    echo "Running on macOS"
    MACOS_VER=$(sw_vers -productVersion | awk -F '.' '{print $1"."$2}')
elif [[ "$OS_TYPE" == "Linux" ]]; then
    echo "Running on Linux"
else
    echo "Unsupported OS: $OS_TYPE"
    exit 1
fi

check_cuda_support() {
    if command -v lspci &> /dev/null; then
        if lspci | grep -i nvidia &> /dev/null; then
            echo "NVIDIA GPU detected. CUDA might be supported."
            return 0
        else
            echo "No NVIDIA GPU detected. CUDA is not supported."
            return 1
        fi
    else
        echo "lspci command not found. Unable to check for NVIDIA GPU."
        return 1
    fi
}
CUDA_SUPPORT=0
if [[ "$OS_TYPE" == "Linux" ]]; then
    check_cuda_support
    CUDA_SUPPORT=$?
    if [[ $CUDA_SUPPORT -ne 0 ]]; then
        echo "CUDA is not supported on this machine. Exiting."
        exit 1
    fi
fi

echo "Downloading SMIRK assets (includes the MediaPipe face_landmarker.task)..."
rm -rf tmp && mkdir tmp && cd tmp
git clone https://github.com/georgeretsi/smirk
mv smirk/assets ../smirk/
cd ..
rm -rf tmp

CONDA_BASE=$(conda info --base)
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate base

ENV_NAME="protego"
PYTHON_VERSION="3.9"
if conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Error: Conda environment '$ENV_NAME' already exists. Please remove it or choose a different name."
    exit 1
fi
echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
echo "Activating conda environment: $ENV_NAME..."
conda activate $ENV_NAME
check_target_env() {
    if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
        echo "Error: Attempting to install packages to '$CONDA_DEFAULT_ENV'. Please remove the automatically created env and set up the environment manually."
        exit 1
    fi
}
check_target_env
echo "Installing packages and downloading SMIRK weights..."
if [[ "$OS_TYPE" == "Linux" ]]; then
    pip install -r requirements.txt
elif [[ "$OS_TYPE" == "Darwin" ]]; then
    pip install -r requirements_mac.txt
fi
conda install zip -y
conda install unzip -y
if [[ "$OS_TYPE" == "Linux" ]]; then
    pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt201/download.html
elif [[ "$OS_TYPE" == "Darwin" ]]; then
    git clone https://github.com/facebookresearch/pytorch3d.git
    cd pytorch3d
    MACOSX_DEPLOYMENT_TARGET=$MACOS_VER CC=clang CXX=clang++ pip install .
    cd ..
    rm -rf pytorch3d
fi
cd smirk
bash quick_install.sh
pip install pytorch_msssim==1.0.0
conda install requests=2.32.3 -y
conda install termcolor=3.1.0 -y
conda install ipython=8.18.1 -y
conda install ipykernel -y
pip install einops
pip install lpips
conda install -c conda-forge ffmpeg -y
cd ..
echo "All packages installed successfully!"

echo "Downloading MTCNN detector weights..."
cd FD_DB/MTCNN/pretrained/
gdown --fuzzy "https://drive.google.com/file/d/1uJopXpkHHzzImZ-4LVWrRHHMbUECi5Fb/view?usp=share_link"
unzip mtcnn_pytorch_weights.zip
rm -f mtcnn_pytorch_weights.zip
cd ../../..

echo "Downloading datasets and checkpoints (FaceScrub eval subset, demo PPT, sample media, ...)..."
echo "Assets without a configured Google Drive link will be skipped (see tools/download_assets.py)."
python -m tools.download_assets || true

echo "ALL DONE!!!"
