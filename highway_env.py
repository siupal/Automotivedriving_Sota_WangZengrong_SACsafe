import gym
import numpy as np
from gym import spaces

class HighwayEnv(gym.Env):
    def __init__(self):
        super(HighwayEnv, self).__init__()
        
        # 根据表2-4设置环境参数
        self.lane_width = 3.75  # 车道宽度(m)
        self.vehicle_length = 5.0  # 车辆长度(m) 
        self.vehicle_width = 1.8  # 车辆宽度(m)
        self.max_speed = 120/3.6  # 最大行驶速度(m/s)
        self.dt = 1.0  # 采样时间(s)
        self.simulation_frequency = 10  # 仿真频率(Hz)
        
        # 设置4条车道
        self.lanes = 4
        self.lane_centers = [(i + 0.5) * self.lane_width for i in range(self.lanes)]
        
        # 定义动作空间和观察空间
        self.action_space = spaces.Box(
            low=np.array([-1, -1]), # [加速度, 转向]
            high=np.array([1, 1]),
            dtype=np.float32
        )
        
        # 观察空间包含:位置(x,y)、速度、航向角等
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(6,),  # [x, y, v, heading, lateral_offset, angular_velocity]
            dtype=np.float32
        )
        
        self.reset()
    
    def reset(self):
        # 初始化车辆状态
        self.state = {
            'x': 0,
            'y': self.lane_centers[1],  # 从第二条车道开始
            'v': 20,  # 初始速度
            'heading': 0,
            'lateral_offset': 0,
            'angular_velocity': 0
        }
        return self._get_obs()
    
    def _get_obs(self):
        return np.array([
            self.state['x'],
            self.state['y'], 
            self.state['v'],
            self.state['heading'],
            self.state['lateral_offset'],
            self.state['angular_velocity']
        ])
    
    def step(self, action):
        # 实现车辆运动学模型
        acc, steering = action
        
        # 更新状态
        dt = 1.0 / self.simulation_frequency
        self.state['v'] += acc * dt
        self.state['v'] = np.clip(self.state['v'], 0, self.max_speed)
        
        # 使用公式(2-32)到(2-36)更新位置和航向
        self.state['x'] += self.state['v'] * np.cos(self.state['heading']) * dt
        self.state['y'] += self.state['v'] * np.sin(self.state['heading']) * dt
        self.state['heading'] += self.state['angular_velocity'] * dt
        
        # 计算奖励
        reward = self._compute_reward()
        
        # 判断是否结束
        done = False
        
        return self._get_obs(), reward, done, {}
    
    def _compute_reward(self):
        # 根据表2-5实现奖励函数
        reward = 0
        
        # 碰撞惩罚
        if self._check_collision():
            reward += -10
            
        # 速度奖励
        v_reward = np.clip(self.state['v'], 0, 12) * (-2/12) + 10/12
        reward += v_reward
        
        # 时间惩罚
        reward += -self.dt/400
        
        return reward
    
    def _check_collision(self):
        # 检查是否超出道路边界
        if self.state['y'] < 0 or self.state['y'] > self.lane_width * self.lanes:
            return True
        return False 