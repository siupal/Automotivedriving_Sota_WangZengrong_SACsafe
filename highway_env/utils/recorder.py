import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List
import json
import os
from datetime import datetime

class TrainingRecorder:
    """训练记录器"""
    def __init__(self, save_dir: str = "results", auto_save_freq: int = 5):
        self.save_dir = save_dir
        self.auto_save_freq = auto_save_freq  # 每隔多少步自动保存
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(save_dir, "curves"), exist_ok=True)
        
        # 尝试加载最新的数据
        self.data = self._load_latest_data()
        if self.data is None:
            self.data = self._init_data()
        
        # 设置matplotlib样式
        try:
            import seaborn as sns
            sns.set_style("whitegrid")  # 使用seaborn的网格样式
        except ImportError:
            # 如果没有seaborn，使用matplotlib的基本样式
            plt.style.use('default')
            plt.grid(True)
        
        # 设置中文字体（如果有的话）
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
            plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        except:
            pass
            
        self.fig = None  # 延迟创建图形
        
        # 添加保存策略参数
        self.snapshot_interval = 10000  # 每10000步保存一次快照
        self.max_snapshots = 10  # 最多保留10个快照
        self.last_snapshot_step = 0  # 上次保存快照的步数
        
    def _init_data(self):
        """初始化数据结构"""
        return {
            "normal": {
                "rewards": [], "lane_changes": [],
                "q_losses": [], "policy_losses": [],
                "mean_speeds": [], "collision_rates": []
            },
            "safe": {
                "rewards": [], "lane_changes": [],
                "q_losses": [], "policy_losses": [],
                "mean_speeds": [], "collision_rates": []
            },
            "steps": [],
            "timestamps": [],
            "last_checkpoint": 0
        }
        
    def _load_latest_data(self):
        """加载最新的数据"""
        try:
            # 查找最新的checkpoint
            checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
            checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")])
            
            if checkpoints:
                latest = os.path.join(checkpoint_dir, checkpoints[-1])
                print(f"Loading checkpoint: {latest}")
                with open(latest, 'r') as f:
                    return json.load(f)
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
        return None
        
    def update(self, step: int, normal_data: Dict, safe_data: Dict):
        """更新训练数据"""
        # 记录时间戳
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.data["timestamps"].append(timestamp)
        self.data["steps"].append(step)
        
        # 记录数据
        for key in normal_data:
            self.data["normal"][key].append(normal_data[key])
        for key in safe_data:
            self.data["safe"][key].append(safe_data[key])
            
        # 自动保存
        if len(self.data["steps"]) % self.auto_save_freq == 0:
            self._save_checkpoint(step)
            
            # 只在重要时刻保存图像
            if (step - self.last_snapshot_step >= self.snapshot_interval or 
                step < 50000):  # 训练初期保存得更频繁
                self._save_curves(step, timestamp)
                self.last_snapshot_step = step
            
        # 打印统计信息
        if len(self.data["steps"]) > 0:
            log_msg = self._format_statistics()
            print(log_msg)
            self._save_log(log_msg)
            
    def _save_checkpoint(self, step: int):
        """保存检查点"""
        # 保存检查点
        checkpoint_path = os.path.join(
            self.save_dir, 
            "checkpoints", 
            f"checkpoint_{step:08d}.json"
        )
        with open(checkpoint_path, 'w') as f:
            json.dump(self.data, f)
            
        # 保存最新数据
        latest_path = os.path.join(self.save_dir, "latest_data.json")
        with open(latest_path, 'w') as f:
            json.dump(self.data, f)
            
        # 删除旧的检查点
        self._cleanup_old_checkpoints()
        
    def _cleanup_old_checkpoints(self, keep_last: int = 5):
        """清理旧的检查点，只保留最新的几个"""
        checkpoint_dir = os.path.join(self.save_dir, "checkpoints")
        checkpoints = sorted([f for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint_")])
        
        # 删除旧的检查点
        for checkpoint in checkpoints[:-keep_last]:
            try:
                os.remove(os.path.join(checkpoint_dir, checkpoint))
            except Exception as e:
                print(f"Error removing old checkpoint: {e}")
        
    def _save_curves(self, step: int, timestamp: str):
        """保存训练曲线"""
        try:
            # 创建新的图形
            plt.figure(figsize=(15, 12))
            
            # 设置全局样式
            plt.rcParams['axes.grid'] = True
            plt.rcParams['grid.alpha'] = 0.3
            plt.rcParams['lines.linewidth'] = 2
            
            # 设置颜色方案
            normal_color = '#FF6B6B'  # 红色
            safe_color = '#4ECDC4'    # 青色
            
            # 1. 奖励曲线
            ax1 = plt.subplot(321)
            ax1.plot(self.data["steps"], self.data["normal"]["rewards"], 
                    color=normal_color, linestyle='--', label='SAC_normal')
            ax1.plot(self.data["steps"], self.data["safe"]["rewards"], 
                    color=safe_color, linestyle='-', label='SAC_safe')
            ax1.set_title('Rewards', fontsize=12, pad=10)
            ax1.set_xlabel('Steps')
            ax1.set_ylabel('Value')
            ax1.legend(frameon=True, fancybox=True, shadow=True)
            
            # 2. 换道次数
            ax2 = plt.subplot(322)
            ax2.plot(self.data["steps"], self.data["normal"]["lane_changes"], 
                    color=normal_color, linestyle='--', label='SAC_normal')
            ax2.plot(self.data["steps"], self.data["safe"]["lane_changes"], 
                    color=safe_color, linestyle='-', label='SAC_safe')
            ax2.set_title('Lane Changes', fontsize=12, pad=10)
            ax2.set_xlabel('Steps')
            ax2.set_ylabel('Count')
            ax2.legend(frameon=True, fancybox=True, shadow=True)
            
            # 3. Q损失
            ax3 = plt.subplot(323)
            q_losses_normal = np.array(self.data["normal"]["q_losses"])
            q_losses_safe = np.array(self.data["safe"]["q_losses"])
            if len(q_losses_normal) > 0 and len(q_losses_safe) > 0:
                min_normal = np.min(q_losses_normal[q_losses_normal > 0]) if np.any(q_losses_normal > 0) else 1e-8
                min_safe = np.min(q_losses_safe[q_losses_safe > 0]) if np.any(q_losses_safe > 0) else 1e-8
                min_loss = max(min(min_normal, min_safe), 1e-8)
                q_losses_normal = np.where(q_losses_normal > 0, q_losses_normal, min_loss)
                q_losses_safe = np.where(q_losses_safe > 0, q_losses_safe, min_loss)
                ax3.plot(self.data["steps"], q_losses_normal, 
                        color=normal_color, linestyle='--', label='SAC_normal')
                ax3.plot(self.data["steps"], q_losses_safe, 
                        color=safe_color, linestyle='-', label='SAC_safe')
                ax3.set_title('Q Losses', fontsize=12, pad=10)
                ax3.set_xlabel('Steps')
                ax3.set_ylabel('Loss')
                ax3.set_yscale('log')
                ax3.legend(frameon=True, fancybox=True, shadow=True)
            
            # 4. 策略损失
            ax4 = plt.subplot(324)
            policy_losses_normal = np.array(self.data["normal"]["policy_losses"])
            policy_losses_safe = np.array(self.data["safe"]["policy_losses"])
            if len(policy_losses_normal) > 0 and len(policy_losses_safe) > 0:
                min_normal = np.min(policy_losses_normal[policy_losses_normal > 0]) if np.any(policy_losses_normal > 0) else 1e-8
                min_safe = np.min(policy_losses_safe[policy_losses_safe > 0]) if np.any(policy_losses_safe > 0) else 1e-8
                min_loss = max(min(min_normal, min_safe), 1e-8)
                policy_losses_normal = np.where(policy_losses_normal > 0, policy_losses_normal, min_loss)
                policy_losses_safe = np.where(policy_losses_safe > 0, policy_losses_safe, min_loss)
                ax4.plot(self.data["steps"], policy_losses_normal, 
                        color=normal_color, linestyle='--', label='SAC_normal')
                ax4.plot(self.data["steps"], policy_losses_safe, 
                        color=safe_color, linestyle='-', label='SAC_safe')
                ax4.set_title('Policy Losses', fontsize=12, pad=10)
                ax4.set_xlabel('Steps')
                ax4.set_ylabel('Loss')
                ax4.set_yscale('log')
                ax4.legend(frameon=True, fancybox=True, shadow=True)
            
            # 5. 平均速度
            ax5 = plt.subplot(325)
            ax5.plot(self.data["steps"], self.data["normal"]["mean_speeds"], 
                    color=normal_color, linestyle='--', label='SAC_normal')
            ax5.plot(self.data["steps"], self.data["safe"]["mean_speeds"], 
                    color=safe_color, linestyle='-', label='SAC_safe')
            ax5.set_title('Mean Speeds', fontsize=12, pad=10)
            ax5.set_xlabel('Steps')
            ax5.set_ylabel('Speed (km/h)')
            ax5.legend(frameon=True, fancybox=True, shadow=True)
            
            # 6. 碰撞率
            ax6 = plt.subplot(326)
            ax6.plot(self.data["steps"], self.data["normal"]["collision_rates"], 
                    color=normal_color, linestyle='--', label='SAC_normal')
            ax6.plot(self.data["steps"], self.data["safe"]["collision_rates"], 
                    color=safe_color, linestyle='-', label='SAC_safe')
            ax6.set_title('Collision Rates', fontsize=12, pad=10)
            ax6.set_xlabel('Steps')
            ax6.set_ylabel('Rate')
            ax6.legend(frameon=True, fancybox=True, shadow=True)
            
            # 调整布局
            plt.tight_layout(pad=3.0)
            
            # 添加总标题
            plt.suptitle(f'Training Progress (Step {step})', 
                        fontsize=14, y=1.02)
            
            # 保存当前图像（覆盖之前的）
            current_fig_path = os.path.join(self.save_dir, "curves", "curves_latest.png")
            plt.savefig(current_fig_path, 
                       dpi=150,
                       bbox_inches='tight',
                       facecolor='white',
                       edgecolor='none')
            
            # 只在特定步数保存快照
            if step % self.snapshot_interval == 0:
                snapshot_path = os.path.join(
                    self.save_dir, 
                    "curves",
                    f"curves_{step//1000}k.png"  # 使用更简洁的文件名
                )
                plt.savefig(snapshot_path,
                           dpi=150,
                           bbox_inches='tight',
                           facecolor='white',
                           edgecolor='none')
                
                # 清理旧的快照
                self._cleanup_snapshots()
            
            plt.close()
            
        except Exception as e:
            print(f"Error saving curves: {e}")
            import traceback
            traceback.print_exc()
            
    def _cleanup_snapshots(self):
        """清理旧的快照，只保留最新的几个"""
        curves_dir = os.path.join(self.save_dir, "curves")
        snapshots = sorted([
            f for f in os.listdir(curves_dir) 
            if f.startswith("curves_") and f.endswith("k.png")
        ])
        
        # 删除多余的快照
        if len(snapshots) > self.max_snapshots:
            for snapshot in snapshots[:-self.max_snapshots]:
                try:
                    os.remove(os.path.join(curves_dir, snapshot))
                except Exception as e:
                    print(f"Error removing old snapshot: {e}")
            
    def _save_csv_data(self):
        """保存CSV格式的训练数据"""
        try:
            import pandas as pd
            
            # 准备数据
            data = {
                'step': self.data["steps"],
                'timestamp': self.data["timestamps"],
                'normal_reward': self.data["normal"]["rewards"],
                'safe_reward': self.data["safe"]["rewards"],
                'normal_lane_changes': self.data["normal"]["lane_changes"],
                'safe_lane_changes': self.data["safe"]["lane_changes"],
                'normal_mean_speed': self.data["normal"]["mean_speeds"],
                'safe_mean_speed': self.data["safe"]["mean_speeds"],
                'normal_collision_rate': self.data["normal"]["collision_rates"],
                'safe_collision_rate': self.data["safe"]["collision_rates"]
            }
            
            # 创建DataFrame并保存
            df = pd.DataFrame(data)
            csv_path = os.path.join(self.save_dir, "training_data.csv")
            df.to_csv(csv_path, index=False)
            
        except Exception as e:
            print(f"Error saving CSV data: {e}")
            
    def _save_log(self, message: str):
        """保存日志"""
        try:
            log_path = os.path.join(self.save_dir, "training.log")
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"{message}\n")
        except Exception as e:
            print(f"Error saving log: {e}")
        
    def _format_statistics(self) -> str:
        """格式化统计信息"""
        stats = [
            "\nCurrent Statistics:",
            f"Steps: {self.data['steps'][-1]}",
            f"Time: {self.data['timestamps'][-1]}",
            "\nRewards:",
            f"SAC_normal: {self.data['normal']['rewards'][-1]:.2f}",
            f"SAC_safe: {self.data['safe']['rewards'][-1]:.2f}",
            "\nLane Changes:",
            f"SAC_normal: {self.data['normal']['lane_changes'][-1]}",
            f"SAC_safe: {self.data['safe']['lane_changes'][-1]}",
            "\nMean Speeds:",
            f"SAC_normal: {self.data['normal']['mean_speeds'][-1]:.1f} km/h",
            f"SAC_safe: {self.data['safe']['mean_speeds'][-1]:.1f} km/h",
            "\nCollision Rates:",
            f"SAC_normal: {self.data['normal']['collision_rates'][-1]:.3f}",
            f"SAC_safe: {self.data['safe']['collision_rates'][-1]:.3f}",
            "\n" + "="*50 + "\n"
        ]
        return "\n".join(stats)