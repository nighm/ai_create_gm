# 检查是否已经在虚拟环境中
if ($env:VIRTUAL_ENV -eq $null) {
    Write-Host "正在激活虚拟环境..."
    
    # 检查虚拟环境是否存在
    if (-not (Test-Path ".venv")) {
        Write-Host "创建新的虚拟环境..."
        python -m venv .venv
    }
    
    # 激活虚拟环境
    .\.venv\Scripts\Activate.ps1
    
    # 更新pip
    python -m pip install --upgrade pip
    
    # 安装依赖
    if (Test-Path "requirements.txt") {
        pip install -r requirements.txt
    } elseif (Test-Path "pyproject.toml") {
        pip install poetry
        poetry install
    }
} else {
    Write-Host "虚拟环境已经激活: $env:VIRTUAL_ENV"
} 