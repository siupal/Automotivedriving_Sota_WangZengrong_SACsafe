import subprocess
import sys
from pathlib import Path

class PycharmSetup:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent
        self.requirements_file = self.project_root / "requirements.txt"
        
    def setup(self):
        """在PyCharm虚拟环境中设置项目"""
        print("开始在PyCharm环境中设置项目...")
        
        # 1. 验证当前环境
        self.verify_venv()
        
        # 2. 安装依赖
        self.install_requirements()
        
        # 3. 安装开发模式的包
        self.install_package()
        
        # 4. 验证安装
        self.verify_installation()
        
        print("环境设置完成!")
        
    def verify_venv(self):
        """验证是否在虚拟环境中"""
        if not hasattr(sys, 'real_prefix') and not sys.base_prefix != sys.prefix:
            print("错误: 请在PyCharm的虚拟环境中运行此脚本")
            sys.exit(1)
            
        print(f"使用Python: {sys.executable}")
        
    def install_requirements(self):
        """安装依赖包"""
        print("\n安装依赖...")
        try:
            # 更新pip
            subprocess.run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"], check=True)
            print("pip更新完成")
            
            # 安装依赖
            subprocess.run([sys.executable, "-m", "pip", "install", "-r", str(self.requirements_file)], check=True)
            print("依赖安装完成")
            
        except subprocess.CalledProcessError as e:
            print(f"安装依赖失败: {e}")
            sys.exit(1)
            
    def install_package(self):
        """以开发模式安装项目包"""
        print("\n以开发模式安装项目包...")
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", "-e", str(self.project_root)], check=True)
            print("项目包安装完成")
        except subprocess.CalledProcessError as e:
            print(f"安装项目包失败: {e}")
            sys.exit(1)
            
    def verify_installation(self):
        """验证安装"""
        print("\n验证安装...")
        
        try:
            # 验证关键包
            import gymnasium
            import numpy
            import torch
            import pygame
            print("基础包导入成功")
            
            # 验证环境
            import highway_env
            env = gymnasium.make('Highway-v0')
            obs, info = env.reset()
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            env.close()
            
            print("环境验证成功!")
            
        except ImportError as e:
            print(f"导入包失败: {e}")
            sys.exit(1)
        except Exception as e:
            print(f"环境验证失败: {e}")
            sys.exit(1)

if __name__ == "__main__":
    setup = PycharmSetup()
    setup.setup() 