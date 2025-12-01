# VideoDescribe Installation Guide

This guide provides instructions for installing VideoDescribe on different platforms.

## Table of Contents
- [Ubuntu/Debian Installation](#ubuntudebian-installation)
- [Windows Installation](#windows-installation)
- [macOS Installation](#macos-installation)
- [Manual Installation](#manual-installation)
- [System Requirements](#system-requirements)
- [Troubleshooting](#troubleshooting)

---

## Ubuntu/Debian Installation

### Automated Installation (Recommended)

We provide an automated installation script for Ubuntu systems:

```bash
# 1. Navigate to the project directory
cd VideoDescribe

# 2. Make the installation script executable
chmod +x install_ubuntu.sh

# 3. Run the installation script
./install_ubuntu.sh

# 4. Activate the virtual environment
source .venv/bin/activate

# 5. Test the installation
python3 main.py --help
```

The script will:
- Install Python 3.12 (if not already installed)
- Install FFmpeg (required for video processing)
- Create a virtual environment
- Install PyTorch with CUDA 13.0 support
- Install all required Python packages
- Verify the installation

---

## Windows Installation

### Prerequisites

1. **Python 3.12**: Download from [python.org](https://www.python.org/downloads/)
2. **FFmpeg**: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
   - Extract FFmpeg and add the `bin` folder to your PATH
   - Or use `winget install ffmpeg` or `choco install ffmpeg`

### Installation Steps

```powershell
# 1. Create virtual environment
python -m venv .venv

# 2. Activate virtual environment
.venv\Scripts\activate

# 3. Upgrade pip
python -m pip install --upgrade pip

# 4. Install PyTorch with CUDA 13.0
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# 5. Install remaining requirements (excluding torch)
pip install transformers accelerate sentencepiece openai-whisper librosa soundfile streamlit numpy tqdm scipy numba decorator audioread pooch joblib scikit-learn lazy-loader msgpack soxr packaging

# 6. Test installation
python main.py --help
```

---

## macOS Installation

### Prerequisites

1. **Homebrew**: Install from [brew.sh](https://brew.sh/)
2. **Python 3.12**:
   ```bash
   brew install python@3.12
   ```
3. **FFmpeg**:
   ```bash
   brew install ffmpeg
   ```

### Installation Steps

```bash
# 1. Create virtual environment
python3.12 -m venv .venv

# 2. Activate virtual environment
source .venv/bin/activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install PyTorch (CPU version for macOS or MPS for Apple Silicon)
# For Apple Silicon (M1/M2/M3):
pip3 install torch torchvision

# 5. Install remaining requirements
pip install -r requirements.txt

# 6. Test installation
python main.py --help
```

---

## Manual Installation

If you prefer to install manually or need a custom setup:

### 1. Install System Dependencies

**FFmpeg** is required for video processing:
- Ubuntu/Debian: `sudo apt-get install ffmpeg`
- Windows: Download from [ffmpeg.org](https://ffmpeg.org/download.html)
- macOS: `brew install ffmpeg`

### 2. Create Virtual Environment

```bash
python3.12 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install PyTorch

Choose the appropriate command based on your system:

**CUDA 13.0 (NVIDIA GPU - Linux/Windows):**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**CUDA 12.1 (NVIDIA GPU - Linux/Windows):**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**CPU Only (No GPU):**
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**macOS (Apple Silicon MPS):**
```bash
pip3 install torch torchvision
```

### 4. Install Remaining Requirements

```bash
pip install -r requirements.txt
```

---

## System Requirements

### Minimum Requirements
- **CPU**: Multi-core processor (Intel i5 or equivalent)
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 10 GB free space for models and dependencies
- **OS**: Ubuntu 20.04+, Windows 10+, or macOS 11+

### Recommended Requirements
- **GPU**: NVIDIA GPU with 8+ GB VRAM (for faster processing)
- **CUDA**: 12.1 or 13.0
- **RAM**: 16 GB or more
- **Storage**: 20 GB free space

### GPU Support
- **NVIDIA GPUs**: Supported via CUDA
- **AMD GPUs**: Limited support (CPU mode recommended)
- **Apple Silicon**: Supported via MPS backend

---

## Troubleshooting

### Python Version Issues
```bash
# Verify Python version
python3.12 --version  # Should be 3.12.x

# If python3.12 is not found, you may need to use python3 or python
python3 --version
```

### FFmpeg Not Found
```bash
# Verify FFmpeg installation
ffmpeg -version
ffprobe -version

# If not found, install FFmpeg:
# Ubuntu: sudo apt-get install ffmpeg
# Windows: Download from ffmpeg.org or use winget/choco
# macOS: brew install ffmpeg
```

### CUDA Issues
```python
# Test CUDA availability
python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# If CUDA is not available but you have an NVIDIA GPU:
# - Verify NVIDIA drivers are installed
# - Reinstall PyTorch with the correct CUDA version
```

### Import Errors
```bash
# If you get import errors, try reinstalling requirements
pip install --force-reinstall -r requirements.txt
```

### Librosa Installation Issues
```bash
# If librosa fails to install, try installing dependencies separately
pip install soundfile scipy numba
pip install librosa
```

### Permission Errors (Linux/macOS)
```bash
# Don't use sudo with pip in a virtual environment
# If you encounter permission errors, ensure virtual environment is activated
source .venv/bin/activate
```

---

## Verifying Installation

After installation, verify that everything is working:

```python
# Test imports
python3 << 'EOF'
import torch
import transformers
import whisper
import librosa
import streamlit
import numpy

print("✓ All packages imported successfully!")
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
EOF
```

Run a test with the example:
```bash
python main.py --help
```

---

## Getting Help

If you encounter issues not covered in this guide:
1. Check the error message carefully
2. Ensure all system dependencies (FFmpeg) are installed
3. Verify Python version is 3.12
4. Try reinstalling in a fresh virtual environment
5. Check GPU drivers if using CUDA

---

## License

This project uses various open-source libraries. Please refer to individual package licenses for more information.
