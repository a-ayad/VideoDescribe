#!/bin/bash

# VideoDescribe Installation Script for Ubuntu
# This script sets up the environment with Python 3.12, PyTorch with CUDA 13.0, and all dependencies

set -e  # Exit on error

echo "======================================"
echo "VideoDescribe Installation Script"
echo "======================================"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[!]${NC} $1"
}

# Check if running on Ubuntu/Debian
if ! command -v apt-get &> /dev/null; then
    print_error "This script is designed for Ubuntu/Debian systems with apt-get"
    exit 1
fi

print_status "Detected Ubuntu/Debian system"

# Update package list
echo ""
echo "Updating package list..."
sudo apt-get update

# Install Python 3.12 if not available
echo ""
echo "Checking Python 3.12 installation..."
if ! command -v python3.12 &> /dev/null; then
    print_warning "Python 3.12 not found. Installing..."

    # Add deadsnakes PPA for Python 3.12
    sudo apt-get install -y software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update

    # Install Python 3.12 and related packages
    sudo apt-get install -y python3.12 python3.12-venv python3.12-dev

    print_status "Python 3.12 installed successfully"
else
    print_status "Python 3.12 is already installed"
fi

# Verify Python 3.12 installation
PYTHON_VERSION=$(python3.12 --version)
print_status "Using: $PYTHON_VERSION"

# Install FFmpeg (required for video processing)
echo ""
echo "Checking FFmpeg installation..."
if ! command -v ffmpeg &> /dev/null; then
    print_warning "FFmpeg not found. Installing..."
    sudo apt-get install -y ffmpeg
    print_status "FFmpeg installed successfully"
else
    print_status "FFmpeg is already installed"
fi

# Verify FFmpeg installation
FFMPEG_VERSION=$(ffmpeg -version | head -n 1)
print_status "Using: $FFMPEG_VERSION"

# Install pip for Python 3.12 if not available
echo ""
echo "Checking pip installation..."
if ! python3.12 -m pip --version &> /dev/null; then
    print_warning "pip not found for Python 3.12. Installing..."
    sudo apt-get install -y python3-pip
    python3.12 -m ensurepip --upgrade
    print_status "pip installed successfully"
else
    print_status "pip is already installed"
fi

# Create virtual environment
echo ""
echo "Creating Python virtual environment..."
if [ -d ".venv" ]; then
    print_warning "Virtual environment already exists. Removing old environment..."
    rm -rf .venv
fi

python3.12 -m venv .venv
print_status "Virtual environment created: .venv"

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source .venv/bin/activate
print_status "Virtual environment activated"

# Upgrade pip, setuptools, and wheel
echo ""
echo "Upgrading pip, setuptools, and wheel..."
pip install --upgrade pip setuptools wheel
print_status "pip, setuptools, and wheel upgraded"

# Install PyTorch with CUDA 13.0 support
echo ""
echo "Installing PyTorch with CUDA 13.0 support..."
echo "This may take a few minutes..."
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
print_status "PyTorch installed successfully"

# Verify PyTorch installation
echo ""
python3 -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

# Install remaining requirements
echo ""
echo "Installing remaining requirements from requirements.txt..."
echo "This may take several minutes..."

# Create a temporary requirements file without torch (already installed)
grep -v "^torch" requirements.txt > requirements_temp.txt

pip install -r requirements_temp.txt
rm requirements_temp.txt

print_status "All requirements installed successfully"

# Final verification
echo ""
echo "======================================"
echo "Installation Summary"
echo "======================================"
echo ""

# Check all critical imports
python3 << 'EOF'
import sys
import importlib

packages = [
    ('torch', 'PyTorch'),
    ('transformers', 'Transformers'),
    ('whisper', 'OpenAI Whisper'),
    ('librosa', 'Librosa'),
    ('streamlit', 'Streamlit'),
    ('numpy', 'NumPy'),
]

print("Package verification:")
all_ok = True
for module_name, display_name in packages:
    try:
        mod = importlib.import_module(module_name)
        version = getattr(mod, '__version__', 'unknown')
        print(f"  ✓ {display_name}: {version}")
    except ImportError as e:
        print(f"  ✗ {display_name}: NOT FOUND")
        all_ok = False

if all_ok:
    print("\n✓ All packages verified successfully!")
else:
    print("\n✗ Some packages failed to import")
    sys.exit(1)
EOF

echo ""
print_status "Installation completed successfully!"
echo ""
echo "======================================"
echo "Next Steps:"
echo "======================================"
echo ""
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Run the video analysis:"
echo "   python3 main.py <video_file.mp4>"
echo ""
echo "3. To deactivate the virtual environment when done:"
echo "   deactivate"
echo ""
echo "======================================"
