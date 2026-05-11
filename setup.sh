#!/bin/bash
# Setup script for KeyFlow Studio development installs on macOS/Linux

set -e

echo "╔════════════════════════════════════════════════════════════╗"
echo "║     KeyFlow Studio - Setup Script                         ║"
echo "╚════════════════════════════════════════════════════════════╝"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Prefer Python 3.11 for compatibility with torch/opencv wheels on macOS.
PYTHON_BIN="python3"
if command -v python3.11 &> /dev/null; then
    PYTHON_BIN="python3.11"
fi

# Check Python version
echo -e "\n${YELLOW}[1/5]${NC} Checking Python version..."
python_version=$($PYTHON_BIN --version 2>&1 | awk '{print $2}')
echo "Found Python ($PYTHON_BIN): $python_version"

# Check FFmpeg
echo -e "\n${YELLOW}[2/5]${NC} Checking FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    ffmpeg_version=$(ffmpeg -version 2>&1 | head -n 1)
    echo "✓ $ffmpeg_version"
else
    echo -e "${RED}✗ FFmpeg not found!${NC}"
    echo "Install with: brew install ffmpeg"
    exit 1
fi

# Create virtual environment
echo -e "\n${YELLOW}[3/5]${NC} Creating virtual environment..."
if [ ! -d ".venv" ]; then
    $PYTHON_BIN -m venv .venv
    echo "✓ Virtual environment created"
else
    echo "✓ Virtual environment already exists"
fi

# Activate virtual environment
source .venv/bin/activate
echo "✓ Virtual environment activated"

# Install Python dependencies
echo -e "\n${YELLOW}[4/5]${NC} Installing Python dependencies..."
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
pip install "numpy<2" "opencv-python==4.11.0.86"
echo "✓ Dependencies installed"

# Optional: Install MatAnyone2 from source
echo -e "\n${YELLOW}[5/5]${NC} MatAnyone2 library..."
echo "To use this application, you need to install MatAnyone2."
echo ""
echo "Option 1: If you have MatAnyone2 locally (RECOMMENDED):"
echo "  pip install -e /path/to/MatAnyone2"
echo ""
echo "Option 2: Install from GitHub:"
echo "  pip install git+https://github.com/pq-yang/MatAnyone2.git"
echo ""
echo "Option 3: Skip for now and install later"
echo ""
read -p "Do you want to install MatAnyone2 now? (y/n/path): " matanyone_response

if [[ "$matanyone_response" == "y" ]]; then
    read -p "Enter path to MatAnyone2 (or press Enter for GitHub): " matanyone_path
    if [ -z "$matanyone_path" ]; then
        echo "Installing from GitHub..."
        if ! pip install git+https://github.com/pq-yang/MatAnyone2.git; then
            echo "Standard install failed. Retrying without optional deps..."
            pip install --no-deps git+https://github.com/pq-yang/MatAnyone2.git
            pip install requests charset-normalizer omegaconf hydra-core einops safetensors huggingface-hub
        fi
    else
        echo "Installing from: $matanyone_path"
        pip install -e "$matanyone_path"
    fi
    echo "✓ MatAnyone2 installed"
elif [[ "$matanyone_response" == "path" ]]; then
    read -p "Enter full path to MatAnyone2: " matanyone_path
    pip install -e "$matanyone_path"
    echo "✓ MatAnyone2 installed"
else
    echo "Skip MatAnyone2 installation"
fi

# Verify installation
echo -e "\n${YELLOW}Verifying installation...${NC}"
python -c "import PySide6; print('✓ PySide6 OK')"
python -c "import torch; print(f'✓ PyTorch {torch.__version__} OK')"
python -c "import cv2; print('✓ OpenCV OK')"
python -c "import imageio; print('✓ imageio OK')"

echo -e "\n${GREEN}╔════════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║              Setup Complete!                              ║${NC}"
echo -e "${GREEN}╚════════════════════════════════════════════════════════════╝${NC}"

echo -e "\n${YELLOW}Next steps:${NC}"
echo "1. Activate the virtual environment:"
echo "   ${GREEN}source .venv/bin/activate${NC}"
echo ""
echo "2. Run the application:"
echo "   ${GREEN}python main.py${NC}"
echo ""
echo "3. (Optional) Create distributable package:"
echo "   Build/distribution tooling is not included in this setup script."
