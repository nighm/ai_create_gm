import os
import sys
import subprocess
import venv
from pathlib import Path

def ensure_venv():
    """确保虚拟环境存在并已激活"""
    venv_path = Path(".venv")
    
    # 检查是否已经在虚拟环境中
    if not os.environ.get("VIRTUAL_ENV"):
        print("未在虚拟环境中，正在处理...")
        
        # 如果虚拟环境不存在，创建它
        if not venv_path.exists():
            print("创建新的虚拟环境...")
            venv.create(".venv", with_pip=True)
        
        # 构建激活命令
        if sys.platform == "win32":
            python_path = str(venv_path / "Scripts" / "python.exe")
        else:
            python_path = str(venv_path / "bin" / "python")
        
        # 使用虚拟环境的Python重新运行当前脚本
        os.environ["VIRTUAL_ENV"] = str(venv_path)
        os.environ["PATH"] = str(venv_path / "Scripts") + os.pathsep + os.environ["PATH"]
        
        # 安装依赖
        subprocess.run([python_path, "-m", "pip", "install", "--upgrade", "pip"])
        subprocess.run([python_path, "-m", "pip", "install", "poetry"])
        subprocess.run([python_path, "-m", "poetry", "install"])
        
        # 重新执行当前命令
        os.execv(python_path, [python_path] + sys.argv)
    else:
        print(f"已在虚拟环境中: {os.environ['VIRTUAL_ENV']}")

def run_game():
    """运行游戏"""
    from src.main import main
    main()

def run_tests():
    """运行测试"""
    import pytest
    pytest.main(["tests/", "-v"])

if __name__ == "__main__":
    # 确保在虚拟环境中
    ensure_venv()
    
    # 设置环境变量
    os.environ["PYTHONPATH"] = "src"
    
    # 根据命令行参数执行不同的功能
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        run_tests()
    else:
        run_game() 