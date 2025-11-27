# Auto Setup Script for Seg Depth (Blackwell / Ada Support)
# Usage: powershell -ExecutionPolicy Bypass -File auto_setup.ps1

$EnvName = "seg_depth_auto"
$PythonVer = "3.11"

Write-Host "===================================================" -ForegroundColor Cyan
Write-Host "   Starting Auto-Setup for $EnvName" -ForegroundColor Cyan
Write-Host "===================================================" -ForegroundColor Cyan

# 1. Create Conda Environment
Write-Host "[1/5] Creating Conda Environment..." -ForegroundColor Yellow
conda env remove -n $EnvName -y 2>$null
conda create -n $EnvName python=$PythonVer pip -y
if ($LASTEXITCODE -ne 0) { Write-Error "Failed to create conda env"; exit 1 }

# Helper to run pip in env
function Run-Pip {
    param([string]$Args)
    # Use cmd /c to ensure correct parsing of arguments by conda run
    conda run -n $EnvName pip $Args
}

# 2. Install PyTorch (User Specified -> Fallback to Nightly)
Write-Host "[2/5] Installing PyTorch..." -ForegroundColor Yellow

# A. Try User's Specific Version (2.9.0+cu128)
Write-Host "   Attempting to install torch==2.9.0 (cu128)..." -ForegroundColor Cyan
# Note: passing args as a single string to helper
conda run -n $EnvName pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 xformers==0.0.33 --index-url https://download.pytorch.org/whl/cu128
$InstallStatus = $LASTEXITCODE

# B. Fallback to Nightly (CUDA 12.4) if specific version fails
if ($InstallStatus -ne 0) {
    Write-Warning "Failed to install torch 2.9.0. Falling back to PyTorch Nightly (CUDA 12.4)..."
    conda run -n $EnvName pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu124
    # Try xformers compatible with nightly (might fail)
    conda run -n $EnvName pip install xformers --no-deps
}

# 3. Install Safe Requirements
Write-Host "[3/5] Installing General Dependencies..." -ForegroundColor Yellow
conda run -n $EnvName pip install -r requirements_safe.txt

# 4. Install Local Packages (Editable Mode)
Write-Host "[4/5] Installing Local Packages (SAM3 & DA3)..." -ForegroundColor Yellow
# We need to run pip install -e . inside the specific directories
# conda run has --cwd argument but it might be newer.
# Let's just use cd and conda run
Set-Location "sam3"
conda run -n $EnvName pip install -e .
Set-Location "..\Depth-Anything-3"
conda run -n $EnvName pip install -e .
Set-Location ".."

# 5. Verification
Write-Host "[5/5] Verifying Installation..." -ForegroundColor Yellow
conda run -n $EnvName pip list | findstr "torch sam3 depth"

Write-Host "===================================================" -ForegroundColor Green
Write-Host "   Setup Complete!" -ForegroundColor Green
Write-Host "   To use: conda activate $EnvName" -ForegroundColor Green
Write-Host "===================================================" -ForegroundColor Green
