#!/bin/bash
# Quick environment + asset setup for Protego.
# Creates a conda env named "protego", installs all dependencies (including SMIRK and PyTorch3D), downloads the third-party detector weights, and then fetches the large datasets/checkpoints.
set -e

ENV_NAME="protego"
PYTHON_VERSION="3.9"
CONDA_BASE=$(conda info --base)
OS_TYPE=$(uname)

# Check OS type and version
if [[ "$OS_TYPE" == "Darwin" ]]; then
    echo "Running on macOS"
    MACOS_VER=$(sw_vers -productVersion | awk -F '.' '{print $1"."$2}')
elif [[ "$OS_TYPE" == "Linux" ]]; then
    echo "Running on Linux"
else
    echo "Unsupported OS: $OS_TYPE"
    exit 1
fi

# Download SMIRK assets (includes the MediaPipe face_landmarker.task)
echo "Downloading SMIRK assets..."
rm -rf tmp && mkdir tmp && cd tmp
git clone https://github.com/georgeretsi/smirk
mv smirk/assets ../smirk/
cd ..
rm -rf tmp

# Set up conda environment and install dependencies
source "$CONDA_BASE/etc/profile.d/conda.sh"
conda activate base
if conda env list | grep -q "^$ENV_NAME\s"; then
    echo "Error: Conda environment '$ENV_NAME' already exists. Please remove it or choose a different name."
    exit 1
fi
echo "Creating conda environment: $ENV_NAME with Python $PYTHON_VERSION..."
conda create -n $ENV_NAME python=$PYTHON_VERSION -y
echo "Activating conda environment: $ENV_NAME..."
conda activate $ENV_NAME
if [[ "$CONDA_DEFAULT_ENV" != "$ENV_NAME" ]]; then
        echo "Error: Attempting to install packages to '$CONDA_DEFAULT_ENV'. Please remove the automatically created env and set up the environment manually."
        exit 1
    fi
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
conda install requests=2.32.3 -y
conda install termcolor=3.1.0 -y
conda install ipython=8.18.1 -y
conda install ipykernel -y
conda install -c conda-forge ffmpeg -y
cd ..
echo "All packages installed successfully!"

# Download Protego assets
echo "Downloading Protego assets..."
cd face_db
gdown --fuzzy "https://drive.google.com/file/d/1j9MOnIXGlElVIHncI9_czFFzRsqnpihR/view?usp=sharing"
unzip face_scrub.zip && rm -f face_scrub.zip
mkdir imgs && cd imgs
gdown --fuzzy "https://drive.google.com/file/d/1LCUWV3BhLBrqHmoq-Yil-4rlsrzxpc8E/view?usp=sharing"
unzip bc_imgs.zip && rm -f bc_imgs.zip
cd ..
mkdir vids && cd vids
gdown --fuzzy "https://drive.google.com/file/d/12pO-xjXa9QAUG63sx-6T-QH4aVxVbZG5/view?usp=sharing"
unzip bc_vids.zip && rm -f bc_vids.zip
cd ../..

cd experiments
gdown --fuzzy "https://drive.google.com/file/d/1Xuj4DWGfudlNCOGP2UojqXxMCS3OOsi0/view?usp=sharing"
unzip default.zip && rm -f default.zip
cd ..

echo "ALL DONE!!!"
