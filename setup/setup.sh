#!/bin/bash

echo "Setting up development environment..."

# 检查 Python 版本
if ! command -v python3.8 &> /dev/null; then
    echo "Python 3.8 not found! Please install Python 3.8 or later."
    exit 1
fi

# 创建并激活虚拟环境
python3.8 -m venv .venv
source .venv/bin/activate

# 更新 pip
python -m pip install --upgrade pip

# 安装依赖
pip install -r requirements.txt

# 安装开发模式的包
pip install -e .

echo "Environment setup complete!" 