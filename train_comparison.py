import gymnasium as gym
import highway_env
import numpy as np
import torch
import matplotlib.pyplot as plt
from train import SAC_normal
from typing import List, Dict
import torch.nn as nn
import os
from tqdm import tqdm
from multiprocessing import Pool
from gymnasium.vector import AsyncVectorEnv
from torch.distributions import Normal
from highway_env.utils.recorder import TrainingRecorder

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# 尝试导入tensorboard
try:
    from torch.utils.tensorboard import SummaryWriter
    use_tensorboard = True
    writer = None
except ImportError:
    print("Tensorboard not available. Training will continue without logging.")
    use_tensorboard = False
    writer = None

class SAC_safe(SAC_normal):
    """SAC_safe算法实现"""
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dim: int = 256,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.4,  # 更大的温度参数使策略更保守
                 batch_size: int = 500,
                 buffer_size: int = int(1e6),
                 update_interval: int = 2):
        """
        初始化SAC_safe算法 (与SAC_normal参数完全相同，只修改温度参数)
        """
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            tau=tau,
            alpha=alpha,  # 传递温度参数
            batch_size=batch_size,
            buffer_size=buffer_size,
            update_interval=update_interval
        )
        
    def compute_risk_correction(self, state: np.ndarray, info: Dict) -> float:
        """计算风险矫正值 (论文公式2-24)"""
        # 获取状态信息
        ego_speed = info.get('speed', 0)  # 主车速度
        front_distance = info.get('front_distance', float('inf'))  # 前车距离
        ttc = front_distance / (ego_speed + 1e-6)  # 碰撞时间(Time To Collision)
        
        # 计算风险值 (基于TTC)
        if ttc < 2.0:  # 高风险
            risk = 1.0
        elif ttc < 4.0:  # 中风险
            risk = 0.5
        else:  # 低风险
            risk = 0.0
            
        # 风险矫正函数
        correction = -risk * (ego_speed / 120.0)  # 速度越快，惩罚越大
        return correction
        
    def update(self, batch_size: int):
        """更新策略"""
        # 采样batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample(batch_size)
            
        # 添加风险矫正
        corrected_rewards = []
        for i in range(len(reward_batch)):
            correction = self.compute_risk_correction(state_batch[i], {})
            corrected_rewards.append(reward_batch[i] + correction)
            
        reward_batch = np.array(corrected_rewards)
        
        # 使用修正后的奖励进行更新
        super().update(batch_size)

def log_performance(episode, normal_reward, safe_reward, normal_changes, safe_changes):
    if use_tensorboard and writer is not None:
        writer.add_scalar('Reward/SAC_normal', normal_reward, episode)
        writer.add_scalar('Reward/SAC_safe', safe_reward, episode)
        writer.add_scalar('Lane_Changes/SAC_normal', normal_changes, episode)
        writer.add_scalar('Lane_Changes/SAC_safe', safe_changes, episode)
        
        # GPU使用情况
        if torch.cuda.is_available():
            memory_allocated = torch.cuda.memory_allocated(0) / 1024**2  # MB
            memory_cached = torch.cuda.memory_cached(0) / 1024**2  # MB
            writer.add_scalar('GPU/Memory_Allocated_MB', memory_allocated, episode)
            writer.add_scalar('GPU/Memory_Cached_MB', memory_cached, episode)
            print(f"\nGPU Memory: {memory_allocated:.1f}MB allocated, {memory_cached:.1f}MB cached")

def train_and_compare():
    """训练和对比两种算法"""
    global writer  # 使用全局writer变量
    
    # 初始化tensorboard writer
    if use_tensorboard:
        writer = SummaryWriter('runs/highway_training')
    
    env = gym.make('Highway-v0', render_mode='human')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 训练参数 (表2-6)
    train_params = {
        'state_dim': state_dim,
        'action_dim': action_dim,
        'hidden_dim': 256,  # 小批量样本数目
        'learning_rate': 3e-4,  # 策略网络学习率
        'gamma': 0.99,  # 折扣系数
        'tau': 0.005,  # 软更新系数
        'batch_size': 500,  # 单个回合最大步数
        'buffer_size': int(1e6),  # 回放缓冲区大小
        'update_interval': 2  # 更新频率
    }
    
    # 创建智能体
    sac_normal = SAC_normal(**train_params)
    sac_safe = SAC_safe(**train_params, alpha=0.4)  # 只修改温度参数
    
    # 训练步数设置 (表2-6)
    initial_episodes = 1000  # 训练前经验采集步数
    total_steps = 200000  # 总训练步数
    simulation_steps = int(40 * 10)  # 40s * 10Hz = 400步/回合
    
    # 记录数据
    normal_rewards: List[float] = []
    safe_rewards: List[float] = []
    normal_lane_changes: List[int] = []
    safe_lane_changes: List[int] = []
    steps_record: List[int] = []
    
    # 初始化训练记录变量
    total_steps_done = 0
    episode = 0
    episode_reward = 0
    lane_changes = 0
    
    # 使用tqdm创建进度条
    pbar = tqdm(total=total_steps, desc="Training Progress")
    
    # 创建记录器
    recorder = TrainingRecorder(auto_save_freq=5)  # 每5步保存一次
    
    # 如果有之前的训练数据，恢复训练状态
    if recorder.data["steps"]:
        last_step = recorder.data["steps"][-1]
        total_steps_done = last_step
        episode = len(recorder.data["steps"])
        print(f"Resuming from step {last_step}, episode {episode}")
    
    try:
        # 创建向量化环境
        n_envs = 8  # 使用8个并行环境
        env_fns = [lambda: gym.make('Highway-v0') for _ in range(n_envs)]
        vec_env = AsyncVectorEnv(env_fns)
        
        # 预训练经验收集 (并行)
        print("Collecting initial experience...")
        states, _ = vec_env.reset()
        
        episodes_per_env = initial_episodes // n_envs
        for _ in tqdm(range(episodes_per_env)):
            # 收集经验
            for step in range(simulation_steps):
                # 获取动作
                actions = [sac_normal.select_action(state, evaluate=True) for state in states]
                
                # 执行动作
                next_states, rewards, dones, _, infos = vec_env.step(actions)
                
                # 存储经验
                for i in range(n_envs):
                    sac_normal.replay_buffer.push(states[i], actions[i], rewards[i], 
                                               next_states[i], dones[i])
                    sac_safe.replay_buffer.push(states[i], actions[i], rewards[i], 
                                             next_states[i], dones[i])
                
                # 更新状态
                states = next_states
                
                # 如果任何环境结束，重置所有环境
                if any(dones):
                    states, _ = vec_env.reset()
                    break
        
        vec_env.close()
        print("Starting training...")
        while total_steps_done < total_steps:
            # 更新环境的训练信息
            env.episode = episode
            env.total_steps = total_steps_done
            env.episode_reward = episode_reward
            env.lane_changes = lane_changes
            
            # 训练SAC_normal
            state, info = env.reset()
            normal_episode_reward = 0
            normal_lane_changes_count = 0
            episode_steps = 0
            normal_speeds = []
            normal_collisions = 0
            
            while True:
                action = sac_normal.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                if env.render_mode == "human":
                    env.render()
                
                done = terminated or truncated
                
                if info.get('lane_change', False):
                    normal_lane_changes_count += 1
                    
                sac_normal.replay_buffer.push(state, action, reward, next_state, done)
                
                if len(sac_normal.replay_buffer) > train_params['batch_size'] and episode_steps % train_params['update_interval'] == 0:
                    sac_normal.update(train_params['batch_size'])
                    
                normal_episode_reward += reward
                state = next_state
                total_steps_done += 1
                episode_steps += 1
                
                # 记录速度和碰撞
                normal_speeds.append(info.get('speed', 0) * 3.6)  # m/s转km/h
                if terminated:
                    normal_collisions += 1
                    
                if done or episode_steps >= simulation_steps:
                    break
                    
            # 计算统计数据
            normal_stats = {
                "rewards": normal_episode_reward,
                "lane_changes": normal_lane_changes_count,
                "q_losses": sac_normal.stats["q_loss"],
                "policy_losses": sac_normal.stats["policy_loss"],
                "mean_speeds": np.mean(normal_speeds),
                "collision_rates": normal_collisions / episode_steps
            }
            
            # 训练SAC_safe
            state, info = env.reset()
            safe_episode_reward = 0
            safe_lane_changes_count = 0
            episode_steps = 0
            safe_speeds = []
            safe_collisions = 0
            
            while True:
                action = sac_safe.select_action(state)
                next_state, reward, terminated, truncated, info = env.step(action)
                if env.render_mode == "human":
                    env.render()
                
                done = terminated or truncated
                
                risk_correction = sac_safe.compute_risk_correction(state, info)
                corrected_reward = reward + risk_correction
                
                if info.get('lane_change', False):
                    safe_lane_changes_count += 1
                    
                sac_safe.replay_buffer.push(state, action, corrected_reward, next_state, done)
                
                if len(sac_safe.replay_buffer) > train_params['batch_size'] and episode_steps % train_params['update_interval'] == 0:
                    sac_safe.update(train_params['batch_size'])
                    
                safe_episode_reward += corrected_reward
                state = next_state
                total_steps_done += 1
                episode_steps += 1
                
                # 记录速度和碰撞
                safe_speeds.append(info.get('speed', 0) * 3.6)  # m/s转km/h
                if terminated:
                    safe_collisions += 1
                    
                if done or episode_steps >= simulation_steps:
                    break
                    
            # 计算统计数据
            safe_stats = {
                "rewards": safe_episode_reward,
                "lane_changes": safe_lane_changes_count,
                "q_losses": sac_safe.stats["q_loss"],
                "policy_losses": sac_safe.stats["policy_loss"],
                "mean_speeds": np.mean(safe_speeds),
                "collision_rates": safe_collisions / episode_steps
            }
            
            # 更新显示用的变量
            episode_reward = (normal_episode_reward + safe_episode_reward) / 2
            lane_changes = normal_lane_changes_count + safe_lane_changes_count
            
            # 打印训练进度
            episode += 1
            print(f"Episode {episode}/{total_steps//simulation_steps}")
            print(f"Steps: {total_steps_done}/{total_steps}")
            print(f"SAC_normal - Reward: {normal_episode_reward:.2f}, Lane Changes: {normal_lane_changes_count}")
            print(f"SAC_safe - Reward: {safe_episode_reward:.2f}, Lane Changes: {safe_lane_changes_count}")
            
            # 每100回合保存模型和绘制曲线
            if episode % 100 == 0:
                # 保存模型
                torch.save({
                    'sac_normal': sac_normal.state_dict(),
                    'sac_safe': sac_safe.state_dict(),
                    'episode': episode,
                    'total_steps': total_steps_done,
                    'normal_rewards': normal_rewards,
                    'safe_rewards': safe_rewards,
                    'normal_lane_changes': normal_lane_changes,
                    'safe_lane_changes': safe_lane_changes
                }, f'models/comparison_{episode}.pth')
                
                # 绘制训练曲线
                plot_training_curves(
                    steps_record,
                    normal_rewards,
                    safe_rewards,
                    normal_lane_changes,
                    safe_lane_changes,
                    episode
                )
                
                # 性能监控
                log_performance(episode, normal_rewards[-1], safe_rewards[-1], 
                              normal_lane_changes[-1], safe_lane_changes[-1])
                
            # 更新进度条
            pbar.update(simulation_steps * 2)  # *2是因为每个episode训练两个智能体
            
            # 更新记录器
            recorder.update(
                step=total_steps_done,
                normal_data=normal_stats,
                safe_data=safe_stats
            )
            
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError occurred: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # 保存最终状态
        recorder._save_checkpoint(total_steps_done)
        env.close()
        if use_tensorboard and writer is not None:
            writer.close()
        torch.cuda.empty_cache()

def plot_training_curves(steps, normal_rewards, safe_rewards, normal_changes, safe_changes, episode):
    """绘制训练曲线"""
    plt.figure(figsize=(12, 5))
    
    # 奖励曲线
    plt.subplot(1, 2, 1)
    plt.plot(steps, normal_rewards, 'r--', label='SAC_normal', alpha=0.8)
    plt.plot(steps, safe_rewards, 'b-', label='SAC_safe', alpha=0.8)
    plt.title('Policy Reward Values')
    plt.xlabel('Training Steps')
    plt.ylabel('Reward')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 换道次数曲线
    plt.subplot(1, 2, 2)
    plt.plot(range(len(normal_changes)), normal_changes, 'r--', label='SAC_normal', alpha=0.8)
    plt.plot(range(len(safe_changes)), safe_changes, 'b-', label='SAC_safe', alpha=0.8)
    plt.title('Number of Lane Changes')
    plt.xlabel('Episode')
    plt.ylabel('Lane Changes')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/comparison_{episode}.png')
    plt.close()
    
    # 实时显示训练信息
    print("\nTraining Progress:")
    print(f"Episode: {episode}")
    print(f"Steps: {steps[-1]}")
    print("\nRewards:")
    print(f"SAC_normal: {normal_rewards[-1]:.2f}")
    print(f"SAC_safe: {safe_rewards[-1]:.2f}")
    print("\nLane Changes:")
    print(f"SAC_normal: {normal_changes[-1]}")
    print(f"SAC_safe: {safe_changes[-1]}")
    print("\n" + "="*50 + "\n")

if __name__ == "__main__":
    # 创建保存目录
    os.makedirs('models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    os.makedirs('runs', exist_ok=True)
    
    train_and_compare() 