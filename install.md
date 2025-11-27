# Seg-Depth Installation Guide (Windows 11)

Dependency management for this project can be tricky due to specific hardware requirements and local package dependencies (SAM3, Depth Anything 3). 

We have provided an automated script to handle everything for you.

### Prerequisites
- Windows 11
- NVIDIA GPU with CUDA support (Driver version supporting CUDA 12.x recommended)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda installed
- PowerShell

### 1. One-Click Installation
This script will automatically:
- Create a clean Conda environment named `seg_depth_auto` (Python 3.11)
- Install PyTorch
- Install all required dependencies
- Compile and install local `sam3` and `Depth-Anything-3` packages

Make sure your dir tree contains:
```
seg_depth/
├── auto_setup.ps1          # One-click installation script (PowerShell)
├── config.yaml             # Configuration file
├── install.md              # Installation guide
├── main.py                 # Main application entry point
├── test_pipline.py         # Pipeline testing script
├── quick_test.py           # Quick diagnostic test script
├── requirements.txt        # Python dependencies list
├── da3_model.py            # Wrapper for Depth Anything 3
├── sam3_model.py           # Wrapper for SAM 3
├── screen_capture.py       # Screen capture module
├── seg_depth_pipeline.py   # Core pipeline logic
├── visualizer.py           # Visualization utilities
├── verify_env.py           # Environment verification script
├── orange_photos/          # Directory for test images
├── Depth-Anything-3/       # [REQUIRED] Local source code for DA3
└── sam3/                   # [REQUIRED] Local source code for SAM3
```

Run the following command in PowerShell:
```powershell
powershell -ExecutionPolicy Bypass -File auto_setup.ps1
```

### 2. Verify Installation
Once the script finishes successfully, activate the environment:

```powershell
conda activate seg_depth_auto
```

#### Check your Python version (should be 3.11.x) and numpy version (should be 1.16.4):
```bash
python --version
conda list | findstr numpy
```

#### You may also manually install addict (at least I ain't pip install it automatically)
```
pip install addict
```

### 3. Run a Quick Test (under \seg_depth):

To verify that SAM3 and Depth Anything 3 are working correctly with your GPU:
```bash
python quick_test.py
```
You should see your PyTorch version, CUDA and GPU status

```bash
python test_pipline.py --folder orange_photos
```

### Troubleshooting
- **`gsplat` Warning**: You may see a warning about `gsplat` dependency. This is for 3D Gaussian Splatting and is **not required** for the core depth/segmentation pipeline. You can safely ignore it.
- **`triton` Warning**: Warnings about `triton` missing are normal on Windows (Triton is Linux-only). Performance will still be excellent using standard CUDA kernels.

