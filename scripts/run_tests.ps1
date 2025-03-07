# 首先运行激活脚本
. .\scripts\activate_venv.ps1

# 设置环境变量
$env:PYTHONPATH = "src"

# 运行测试
python -m pytest tests/ -v 