# setup_clean_env.ps1
# 自动化环境配置脚本

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "   Seg-Depth Environment Setup Script"
Write-Host "===============================================" -ForegroundColor Cyan

# 1. 检查 Conda 环境
$envName = $env:CONDA_DEFAULT_ENV
Write-Host "`nCurrent Conda Environment: $envName" -ForegroundColor Yellow
if ($envName -eq "base") {
    Write-Error "Error: You are in 'base' env. Please create and activate a new env first!"
    Write-Host "Run: conda create -n seg_depth_clean python=3.10 -y"
    Write-Host "Run: conda activate seg_depth_clean"
    exit 1
}

# 2. 安装 PyTorch 和 xformers (针对 RTX 6000 Blackwell)
Write-Host "`n[1/4] Installing PyTorch 2.9 & xformers..." -ForegroundColor Green
# 使用你指定的 2.9.0 版本 (假设源有效)
pip install torch==2.9.0 torchvision==0.24.0 torchaudio==2.9.0 xformers==0.0.33 --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { Write-Error "PyTorch install failed"; exit 1 }

# 3. 安装通用依赖
Write-Host "`n[2/4] Installing requirements..." -ForegroundColor Green
pip install -r requirements_clean.txt
if ($LASTEXITCODE -ne 0) { Write-Error "Requirements install failed"; exit 1 }

# 4. 安装本地库 (SAM3 & DA3)
Write-Host "`n[3/4] Installing Local Packages (SAM3 & DA3)..." -ForegroundColor Green

if (Test-Path "sam3") {
    Write-Host "  Installing SAM3..."
    cd sam3
    pip install -e .
    cd ..
} else {
    Write-Error "Folder 'sam3' not found!"
}

if (Test-Path "Depth-Anything-3") {
    Write-Host "  Installing Depth-Anything-3..."
    cd Depth-Anything-3
    pip install -e .
    cd ..
} else {
    Write-Error "Folder 'Depth-Anything-3' not found!"
}

# 5. 验证
Write-Host "`n[4/4] Verifying Installation..." -ForegroundColor Green
python -c "import torch; print(f'PyTorch: {torch.__version__}'); import sam3; import depth_anything_3; print('All modules imported successfully!')"

Write-Host "`n===============================================" -ForegroundColor Cyan
Write-Host "   Setup Complete! Ready to test."
Write-Host "===============================================" -ForegroundColor Cyan

