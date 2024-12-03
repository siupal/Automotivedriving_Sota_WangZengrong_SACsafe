import gymnasium as gym
import highway_env
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal
import random
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class ReplayBuffer:
    """经验回放池"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List = []
        self.position = 0
        
    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity
        
    def sample(self, batch_size: int) -> Tuple:
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.stack, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self) -> int:
        return len(self.buffer)

class QNetwork(nn.Module):
    """Q网络"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(QNetwork, self).__init__()
        
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        ).to(device)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        sa = torch.cat([state, action], 1)
        return self.q1(sa), self.q2(sa)

class PolicyNetwork(nn.Module):
    """策略网络"""
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int):
        super(PolicyNetwork, self).__init__()
        
        # 计算输入维度: 主车6维 + 7辆车各6维 = 48维
        input_dim = 6 + 7 * 6  # 48维
        
        # 使用更好的初始化方法
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU()
        ).to(device)
        
        # 初始化均值和标准差网络
        self.mean = nn.Linear(hidden_dim, action_dim).to(device)
        self.log_std = nn.Linear(hidden_dim, action_dim).to(device)
        
        # 使用正交初始化
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
                nn.init.constant_(layer.bias, 0)
        
        nn.init.orthogonal_(self.mean.weight, gain=0.01)
        nn.init.constant_(self.mean.bias, 0)
        nn.init.orthogonal_(self.log_std.weight, gain=0.01)
        nn.init.constant_(self.log_std.bias, 0)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, -20, 2)  # 限制标准差范围
        return mean, log_std

class SAC_normal:
    """SAC算法实现"""
    def __init__(self, 
                 state_dim: int, 
                 action_dim: int, 
                 hidden_dim: int = 256,
                 learning_rate: float = 3e-4,
                 gamma: float = 0.99,
                 tau: float = 0.005,
                 alpha: float = 0.2,
                 batch_size: int = 500,
                 buffer_size: int = int(1e6),
                 update_interval: int = 2):
        """
        初始化SAC算法 (表2-6参数)
        
        Args:
            state_dim: 状态空间维度
            action_dim: 动作空间维度
            hidden_dim: 隐藏层维度 (256)
            learning_rate: 学习率 (3e-4)
            gamma: 折扣因子 (0.99)
            tau: 软更新系数 (0.005)
            alpha: 温度参数 (0.2)
            batch_size: 批次大小 (500)
            buffer_size: 经验池大小 (1e6)
            update_interval: 更新频率 (2)
        """
        # 网络结构
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.q_net = QNetwork(state_dim, action_dim, hidden_dim)
        self.target_q_net = QNetwork(state_dim, action_dim, hidden_dim)
        
        # 复制参数到目标网络
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(param.data)
            
        # 优化器 (Adam)
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.q_optimizer = optim.Adam(self.q_net.parameters(), lr=learning_rate)
        
        # 超参数
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.batch_size = batch_size
        self.update_interval = update_interval
        
        # 经验回放池
        self.replay_buffer = ReplayBuffer(buffer_size)
        
        # 记录训练数据
        self.rewards: List[float] = []
        self.lane_changes: List[int] = []
        
        # 添加统计数据
        self.stats = {
            "q_loss": 0.0,
            "policy_loss": 0.0,
            "mean_speed": 0.0,
            "collision_rate": 0.0
        }
        
    def select_action(self, state: np.ndarray, evaluate: bool = False) -> np.ndarray:
        """选择动作"""
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        mean, log_std = self.policy(state)
        std = log_std.exp()
        
        if evaluate:
            return mean.cpu().detach().numpy()[0]
            
        dist = Normal(mean, std)
        action = dist.rsample()
        return action.cpu().detach().numpy()[0]
        
    def update(self, batch_size: int):
        """更新策略"""
        # 采样batch
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = \
            self.replay_buffer.sample(batch_size)
            
        # 转换为tensor并移到GPU
        state_batch = torch.FloatTensor(state_batch).to(device)
        action_batch = torch.FloatTensor(action_batch).to(device)
        reward_batch = torch.FloatTensor(reward_batch).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(next_state_batch).to(device)
        done_batch = torch.FloatTensor(done_batch).unsqueeze(1).to(device)
        
        # 计算目标Q值
        with torch.no_grad():
            next_state_action, next_state_log_pi = self._sample_action(next_state_batch)
            next_q1_target, next_q2_target = self.target_q_net(next_state_batch, next_state_action)
            next_q_target = torch.min(next_q1_target, next_q2_target)
            next_q_value = reward_batch + (1 - done_batch) * self.gamma * \
                          (next_q_target - self.alpha * next_state_log_pi.unsqueeze(1))
                          
        # 更新Q网络
        q1, q2 = self.q_net(state_batch, action_batch)
        q1_loss = nn.MSELoss()(q1, next_q_value.detach())
        q2_loss = nn.MSELoss()(q2, next_q_value.detach())
        q_loss = q1_loss + q2_loss
        
        self.q_optimizer.zero_grad()
        q_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), max_norm=1.0)
        self.q_optimizer.step()
        
        # 更新策略网络
        pi, log_pi = self._sample_action(state_batch)
        q1_pi, q2_pi = self.q_net(state_batch, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        policy_loss = (self.alpha * log_pi - min_q_pi).mean()
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        self.policy_optimizer.step()
        
        # 记录统计数据
        self.stats["q_loss"] = q_loss.item()
        self.stats["policy_loss"] = policy_loss.item()
        
        # 软更新目标网络
        for target_param, param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            
    def _sample_action(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """采样动作"""
        mean, log_std = self.policy(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        x_t = dist.rsample()
        log_prob = dist.log_prob(x_t)
        return x_t, log_prob
    
    def state_dict(self):
        """获取模型状态"""
        return {
            'policy': self.policy.state_dict(),
            'q_net': self.q_net.state_dict(),
            'target_q_net': self.target_q_net.state_dict(),
            'policy_optimizer': self.policy_optimizer.state_dict(),
            'q_optimizer': self.q_optimizer.state_dict()
        }
    
    def load_state_dict(self, state_dict):
        """加载模型状态"""
        self.policy.load_state_dict(state_dict['policy'])
        self.q_net.load_state_dict(state_dict['q_net'])
        self.target_q_net.load_state_dict(state_dict['target_q_net'])
        self.policy_optimizer.load_state_dict(state_dict['policy_optimizer'])
        self.q_optimizer.load_state_dict(state_dict['q_optimizer'])

def train():
    """训练主函数"""
    env = gym.make('Highway-v0')
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    # 创建智能体
    agent = SAC_normal(state_dim, action_dim)
    
    # 训练参数
    episodes = 1000
    max_steps = 40
    update_interval = 2
    
    for episode in range(episodes):
        state, _ = env.reset()
        episode_reward = 0
        lane_changes = 0
        
        for step in range(max_steps):
            # 选择动作
            action = agent.select_action(state)
            
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            # 记录换道次数
            if info.get('lane_change', False):
                lane_changes += 1
            
            # 存储经验
            agent.replay_buffer.push(state, action, reward, next_state, done)
            
            # 更新策略
            if len(agent.replay_buffer) > agent.batch_size and step % update_interval == 0:
                agent.update(agent.batch_size)
            
            episode_reward += reward
            state = next_state
            
            if done:
                break
        
        # 记录训练数据
        agent.rewards.append(episode_reward)
        agent.lane_changes.append(lane_changes)
        
        print(f"Episode {episode}, Reward: {episode_reward}, Lane Changes: {lane_changes}")
        
        # 保存模型和绘制曲线
        if episode % 100 == 0:
            torch.save({
                'policy': agent.policy.state_dict(),
                'q_net': agent.q_net.state_dict(),
                'target_q_net': agent.target_q_net.state_dict()
            }, f'models/sac_normal_{episode}.pth')
            
            # 绘制训练曲线
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.plot(agent.rewards)
            plt.title('Rewards')
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            
            plt.subplot(1, 2, 2)
            plt.plot(agent.lane_changes)
            plt.title('Lane Changes')
            plt.xlabel('Episode')
            plt.ylabel('Number of Lane Changes')
            
            plt.savefig(f'results/training_curves_{episode}.png')
            plt.close()

if __name__ == "__main__":
    train() 