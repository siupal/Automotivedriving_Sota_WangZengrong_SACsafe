@echo off
echo Setting up development environment...

:: 检查 Python 版本
python --version 2>NUL
if errorlevel 1 (
    echo Python not found! Please install Python 3.8 or later.
    exit /b 1
)

:: 创建并激活虚拟环境
python -m venv .venv
call .venv\Scripts\activate

:: 更新 pip
python -m pip install --upgrade pip

:: 安装依赖
pip install -r requirements.txt

:: 安装开发模式的包
pip install -e .

echo Environment setup complete!
pause 