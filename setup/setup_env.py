import os
import subprocess
import sys
import platform
from pathlib import Path

class EnvironmentSetup:
    def __init__(self):
        self.python_version = "3.8"
        self.project_name = "highway_env"
        self.requirements_file = "requirements.txt"
        
    def setup(self):
        """设置完整的开发环境"""
        print("开始设置开发环境...")
        
        # 1. 创建虚拟环境
        self.create_virtual_env()
        
        # 2. 激活虚拟环境
        self.activate_virtual_env()
        
        # 3. 安装依赖
        self.install_requirements()
        
        # 4. 验证安装
        self.verify_installation()
        
        print("环境设置完成!")
        
    def create_virtual_env(self):
        """创建虚拟环境"""
        print("创建虚拟环境...")
        
        if platform.system() == "Windows":
            python_cmd = f"python{self.python_version}"
        else:
            python_cmd = f"python{self.python_version}"
            
        try:
            subprocess.run([python_cmd, "-m", "venv", ".venv"], check=True)
        except subprocess.CalledProcessError:
            print(f"创建虚拟环境失败，请确保已安装 Python {self.python_version}")
            sys.exit(1)
            
    def activate_virtual_env(self):
        """激活虚拟环境"""
        print("激活虚拟环境...")
        
        if platform.system() == "Windows":
            activate_script = ".venv\\Scripts\\activate"
        else:
            activate_script = ".venv/bin/activate"
            
        if not os.path.exists(activate_script):
            print("虚拟环境激活脚本不存在")
            sys.exit(1)
            
        # 设置环境变量
        os.environ["VIRTUAL_ENV"] = str(Path(".venv").absolute())
        os.environ["PATH"] = str(Path(".venv/Scripts").absolute() if platform.system() == "Windows" 
                                else Path(".venv/bin").absolute()) + os.pathsep + os.environ["PATH"]
        
    def install_requirements(self):
        """安装依赖"""
        print("安装依赖...")
        
        try:
            # 更新pip
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            
            # 安装依赖
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", self.requirements_file], check=True)
            
            # 安装开发模式的包
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", "."], check=True)
            
        except subprocess.CalledProcessError as e:
            print(f"安装依赖失败: {e}")
            sys.exit(1)
            
    def verify_installation(self):
        """验证安装"""
        print("验证安装...")
        
        try:
            # 尝试导入关键包
            import gymnasium
            import numpy
            import torch
            import pygame
            
            # 验证环境
            import highway_env
            env = gymnasium.make('Highway-v0')
            env.reset()
            env.step(env.action_space.sample())
            env.close()
            
            print("验证成功!")
            
        except ImportError as e:
            print(f"导入包失败: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"环境验证失败: {e}")
            sys.exit(1)

if __name__ == "__main__":
    setup = EnvironmentSetup()
    setup.setup() 