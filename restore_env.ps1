# restore_env.ps1
# 使用 environment.yml 还原完美环境

Write-Host "===============================================" -ForegroundColor Cyan
Write-Host "   Seg-Depth Environment Restore Script"
Write-Host "===============================================" -ForegroundColor Cyan

# 1. 从 environment.yml 创建/更新环境
Write-Host "`n[1/3] Restoring Conda Environment from environment.yml..." -ForegroundColor Green
# 如果环境已存在，这会更新它；如果不存在，会创建它
conda env update -n seg_depth_torch_2_9_0 --file environment.yml --prune
if ($LASTEXITCODE -ne 0) { Write-Error "Conda env update failed"; exit 1 }

# 2. 安装本地库 (SAM3 & DA3)
# environment.yml 不包含本地 pip install -e . 的包，必须手动补上
Write-Host "`n[2/3] Installing Local Packages (SAM3 & DA3)..." -ForegroundColor Green

# 激活环境的 context (这在脚本中比较 tricky，通常需要用户手动激活)
# 我们尝试直接使用该环境的 python 解释器路径（如果能找到）
# 或者假设用户稍后会激活。但为了确保 pip install -e . 生效，最好是在激活的环境中运行。

Write-Host "IMPORTANT: Please make sure you have activated the environment:" -ForegroundColor Yellow
Write-Host "conda activate seg_depth_torch_2_9_0" -ForegroundColor Yellow
Write-Host "Then press Enter to continue installation of local packages..." -ForegroundColor Yellow
Read-Host "Press Enter"

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

# 3. 验证
Write-Host "`n[3/3] Verifying Installation..." -ForegroundColor Green
python -c "import torch; print(f'PyTorch: {torch.__version__}'); import sam3; import depth_anything_3; print('All modules imported successfully!')"

Write-Host "`n===============================================" -ForegroundColor Cyan
Write-Host "   Restore Complete! Ready to work."
Write-Host "===============================================" -ForegroundColor Cyan

