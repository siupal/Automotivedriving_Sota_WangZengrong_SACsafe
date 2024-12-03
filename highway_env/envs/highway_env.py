from highway_env.utils.config import EnvConfig
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, List, Optional, Tuple
from highway_env.vehicle.kinematics import Vehicle

class HighwayEnv(gym.Env):
    """
    Highway driving environment.
    
    基于论文要求实现的4车道高速公路环境：
    - 车道宽度: 3.75m
    - 车辆长度: 5.0m
    - 车辆宽度: 1.8m
    - 最大速度: 120km/h
    - 采样时间: 1s
    - 仿真频率: 10Hz
    """
    
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps': 15
    }
    
    def __init__(self, render_mode=None):
        super(HighwayEnv, self).__init__()
        
        self.config = EnvConfig()
        
        # 创建道路 (表2-4)
        self.road_length = 1000  # 道路长度(m)
        self.lanes = 4  # 4车道
        self.lane_width = 3.75  # 车道宽度(m)
        
        # 创建车辆列表
        self.vehicles: List[Vehicle] = []
        self.controlled_vehicle: Optional[Vehicle] = None
        
        # 观察空间
        obs_shape = (self.config.observation.n_observation_vehicles * 
                    len(self.config.observation.observation_features),)
        self.observation_space = spaces.Box(
            low=-1000,
            high=1000,
            shape=obs_shape,
            dtype=np.float32
        )
        
        # 动作空间: [加速度, 转向角]
        self.action_space = spaces.Box(
            low=np.array([-5.0, -np.pi/4], dtype=np.float32),  # 最小加速度和转向角
            high=np.array([5.0, np.pi/4], dtype=np.float32),   # 最大加速度和转向角
            dtype=np.float32
        )
        
        # 仿真参数 (表2-4)
        self.dt = 0.1  # 仿真步长(s)
        self.simulation_frequency = 10  # 仿真频率(Hz)
        self.sampling_time = 1.0  # 采样时间(s)
        self.max_episode_steps = int(40 * self.simulation_frequency)  # 40s * 10Hz = 400步
        
        # 渲染设置
        self.render_mode = render_mode
        self.screen = None
        self.clock = None
        
        # 时间计数器
        self.time = 0
        self.steps = 0
        
    def _create_vehicles(self) -> None:
        """创建初始车辆，按照图2-4布局但更密集"""
        # 创建主车 (绿色)：在第二条车道
        ego_config = {
            "length": 5.0,  # 车长5.0m
            "width": 1.8,   # 车宽1.8m
            "initial_speed": 90/3.6,  # 初始速度90km/h
            "lane_id": 1,  # 从第二车道开始
            "position": [50, self.config.road.lane_centers[1]],  # 靠前位置
            "controlled": True
        }
        self.controlled_vehicle = self._create_vehicle(ego_config)
        self.vehicles.append(self.controlled_vehicle)
        
        # 车道速度限制 (从下到上车道依次增大)
        lane_speeds = {
            0: (80, 85),    # 最右车道: 80-85km/h
            1: (85, 90),    # 次右车道: 85-90km/h
            2: (90, 95),    # 次左车道: 90-95km/h
            3: (95, 100)    # 最左车道: 95-100km/h
        }
        
        # 每个车道的车辆数量
        vehicles_per_lane = {
            0: 4,  # 最右车道4辆
            1: 3,  # 次右车道3辆
            2: 3,  # 次左车道3辆
            3: 4   # 最左车道4辆
        }
        
        # 在每个车道上生成车辆
        for lane_id, num_vehicles in vehicles_per_lane.items():
            # 计算车道上的均匀分布位置
            min_gap = 30  # 最小车间距(m)
            max_gap = 50  # 最大车间距(m)
            
            # 生成基础位置
            base_positions = np.linspace(0, self.road_length-100, num_vehicles)
            
            # 添加随机扰动
            positions = []
            for pos in base_positions:
                # 在基础位置附近添加随机扰动
                new_pos = pos + np.random.uniform(-10, 10)
                # 确保与已有车辆保持最小距离
                while any(abs(p - new_pos) < min_gap for p in positions):
                    new_pos = pos + np.random.uniform(-10, 10)
                positions.append(new_pos)
            
            # 获取该车道的速度范围
            min_speed, max_speed = lane_speeds[lane_id]
            
            # 创建车辆
            for x_pos in positions:
                # 根据位置调整速度，使后车速度略快
                relative_pos = x_pos / self.road_length
                speed_range = (max_speed - min_speed) * relative_pos
                vehicle_speed = min_speed + speed_range
                
                vehicle_config = {
                    "length": 5.0,
                    "width": 1.8,
                    "initial_speed": vehicle_speed/3.6,  # km/h转m/s
                    "lane_id": lane_id,
                    "position": [x_pos, self.config.road.lane_centers[lane_id]],
                    "controlled": False,
                    "idm_params": {
                        "a_max": 5.0,      # 最大加速度 ω = 5.0 m/s²
                        "b_comfort": -4.0,  # 最大减速度 b = -4.0 m/s²
                        "time_headway": 1.5,  # 安全时间间隔 T = 1.5 s
                        "distance_min": 10.0,  # 最小相对距离 d₀ = 10.0 m
                        "delta": 4.0,      # 加速度指数 δ = 4.0
                    },
                    "mobil_params": {
                        "politeness": 0.5,  # 礼让因子
                        "safe_threshold": -4.0,  # 安全阈值(m/s²)
                        "acceleration_threshold": 0.1,  # 换道阈值(m/s²)
                    }
                }
                vehicle = self._create_vehicle(vehicle_config)
                self.vehicles.append(vehicle)
        
    def _create_vehicle(self, config: Dict) -> Vehicle:
        """创建单个车辆"""
        # 确保配置中包含必要的参数
        if not config["controlled"]:
            if "idm_params" not in config:
                config["idm_params"] = {
                    "a_max": 5.0,      # 最大加速度 ω = 5.0 m/s²
                    "b_comfort": -4.0,  # 最大减速度 b = -4.0 m/s²
                    "time_headway": 1.5,  # 安全时间间隔 T = 1.5 s
                    "distance_min": 10.0,  # 最小相对距离 d₀ = 10.0 m
                    "delta": 4.0,      # 加速度指数 δ = 4.0
                }
            if "mobil_params" not in config:
                config["mobil_params"] = {
                    "politeness": 0.5,
                    "safe_threshold": -4.0,
                    "acceleration_threshold": 0.1,
                }
        
        vehicle = Vehicle(config)
        return vehicle
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 重置时间
        self.time = 0
        self.steps = 0
        
        # 重置车辆
        self.vehicles.clear()
        self.controlled_vehicle = None
        self._create_vehicles()
        
        # 获取观察
        observation = self._get_obs()
        info = {
            'time': self.time,
            'steps': self.steps
        }
        
        return observation, info
        
    def step(self, action):
        """执行一步环境交互"""
        # 更新时间
        self.time += self.config.simulation.dt
        self.steps += 1
        
        # 更新车辆状态
        self.controlled_vehicle.step(self.config.simulation.dt, action, env=self)
        for vehicle in self.vehicles:
            if vehicle != self.controlled_vehicle:
                vehicle.step(self.config.simulation.dt, env=self)
        
        # 获取观察、奖励和终止信息
        observation = self._get_obs()
        reward = self._compute_reward()
        terminated = self._is_terminated()
        truncated = self.time >= self.config.simulation.duration
        
        # 检查换道
        lane_changed = False
        if hasattr(self.controlled_vehicle, 'last_lane_id'):
            if self.controlled_vehicle.lane_id != self.controlled_vehicle.last_lane_id:
                lane_changed = True
        self.controlled_vehicle.last_lane_id = self.controlled_vehicle.lane_id
        
        info = {
            'time': self.time,
            'steps': self.steps,
            'speed': self.controlled_vehicle.velocity[0],
            'lane_change': lane_changed
        }
        
        # 只在训练阶段渲染，收集阶段跳过
        if self.render_mode == "human" and not hasattr(self, '_collecting'):
            self.render()
        
        return observation, reward, terminated, truncated, info
        
    def _get_obs(self):
        """获取观察空间"""
        # 获取主车状态
        ego_state = np.array([
            self.controlled_vehicle.position[0],  # x位置
            self.controlled_vehicle.position[1],  # y位置
            self.controlled_vehicle.velocity[0],  # x方向速度
            self.controlled_vehicle.velocity[1],  # y方向速度
            self.controlled_vehicle.heading,      # 航向角
            self.controlled_vehicle.lane_id,      # 当前车道
        ], dtype=np.float32)

        # 获取其他车辆的状态
        other_vehicles = []
        for v in self.vehicles:
            if v != self.controlled_vehicle:
                # 计算相对位置和速度
                rel_pos = v.position - self.controlled_vehicle.position
                rel_vel = v.velocity - self.controlled_vehicle.velocity
                
                # 只记录前后左右的车辆
                if np.linalg.norm(rel_pos) <= self.config.observation.observation_range:
                    other_vehicles.append([
                        rel_pos[0],  # 相对x位置
                        rel_pos[1],  # 相对y位置
                        rel_vel[0],  # 相对x速度
                        rel_vel[1],  # 相对y速度
                        v.heading - self.controlled_vehicle.heading,  # 相对航向角
                        v.lane_id - self.controlled_vehicle.lane_id,  # 相对车道
                    ])

        # 填充固定数量的车辆信息
        max_vehicles = 7  # 最多观察7辆车
        while len(other_vehicles) < max_vehicles:
            other_vehicles.append([0, 0, 0, 0, 0, 0])  # 用0填充
        other_vehicles = other_vehicles[:max_vehicles]  # 截断多余的车辆

        # 合并所有观察
        observation = np.concatenate([ego_state] + other_vehicles)
        return observation.astype(np.float32)
    
    def render(self):
        """渲染环境，按照图2-4样式"""
        if self.render_mode is None:
            return None
        
        try:
            import pygame
            
            # 渲染参数
            PIXELS_PER_METER = 10  # 每米对应的像素数
            ROAD_LENGTH_PIXELS = 1200  # 道路长度(像素)
            INFO_WIDTH = 300  # 增加信息面板宽度
            LANE_WIDTH_PIXELS = int(3.75 * PIXELS_PER_METER)  # 车道宽度(像素)
            SCREEN_WIDTH = ROAD_LENGTH_PIXELS + INFO_WIDTH
            SCREEN_HEIGHT = int(4 * LANE_WIDTH_PIXELS)  # 4车道总高度
            
            if self.screen is None:
                pygame.init()
                if self.render_mode == "human":
                    pygame.display.init()
                    self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
                    pygame.display.set_caption("Highway Environment")
                    self.clock = pygame.time.Clock()
                    self.font = pygame.font.Font(None, 24)
                else:
                    self.screen = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT))
            
            # 处理事件
            if self.render_mode == "human":
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.close()
                        return None
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            self.close()
                            return None
            
            # 绘制背景
            self.screen.fill((255, 255, 255))  # 白色背景
            
            # 绘道路
            road_surface = pygame.Surface((ROAD_LENGTH_PIXELS, SCREEN_HEIGHT))
            road_surface.fill((220, 220, 220))  # 浅灰色道路
            self.screen.blit(road_surface, (0, 0))
            
            # 绘制车道线
            for i in range(5):  # 4车道需要5条分隔线
                y = i * LANE_WIDTH_PIXELS
                if i == 0 or i == 4:  # 最上和最下的实线
                    pygame.draw.line(self.screen, (100, 100, 100), (0, y), (ROAD_LENGTH_PIXELS, y), 3)
                else:  # 中间的虚线
                    for x in range(0, ROAD_LENGTH_PIXELS, 30):  # 每30像素画一段虚线
                        pygame.draw.line(self.screen, (100, 100, 100), (x, y), (x + 15, y), 2)
            
            # 绘制车辆
            for vehicle in self.vehicles:
                # 计算屏幕坐标
                x = (vehicle.position[0] % self.road_length) * ROAD_LENGTH_PIXELS / self.road_length
                y = (vehicle.lane_id + 0.5) * LANE_WIDTH_PIXELS
                
                # 计算车辆尺寸
                vehicle_length = vehicle.length * PIXELS_PER_METER
                vehicle_width = vehicle.width * PIXELS_PER_METER
                
                # 绘制车辆矩形
                color = (0, 200, 0) if vehicle.controlled else (0, 0, 200)
                rect = pygame.Rect(
                    x - vehicle_length/2,
                    y - vehicle_width/2,
                    vehicle_length,
                    vehicle_width
                )
                pygame.draw.rect(self.screen, color, rect)
            
            # 绘制信息面板
            info_x = ROAD_LENGTH_PIXELS + 10
            info_y = 10
            line_height = 25
            
            # 绘制分隔线
            pygame.draw.line(self.screen, (0, 0, 0), 
                            (ROAD_LENGTH_PIXELS, 0), 
                            (ROAD_LENGTH_PIXELS, SCREEN_HEIGHT), 2)
            
            # 1. 环境参数 (Environment Parameters)
            title = self.font.render("Environment Parameters", True, (0, 0, 0))
            self.screen.blit(title, (info_x, info_y))
            info_y += line_height * 1.5
            
            basic_info = [
                ("Lane Width", "3.75 m"),
                ("Vehicle Length", "5.0 m"),
                ("Vehicle Width", "1.8 m"),
                ("Max Speed", "120 km/h"),
                ("Sample Time", "1.0 s")
            ]
            
            for param, value in basic_info:
                text = self.font.render(f"{param}: {value}", True, (0, 0, 0))
                self.screen.blit(text, (info_x, info_y))
                info_y += line_height
            
            info_y += line_height
            
            # 2. 训练信息 (Training Status)
            title = self.font.render("Training Status", True, (0, 0, 0))
            self.screen.blit(title, (info_x, info_y))
            info_y += line_height * 1.5
            
            # 获取当前算法类型
            algorithm = "SAC_safe" if hasattr(self, "compute_risk_correction") else "SAC_normal"
            
            training_info = [
                ("Algorithm", algorithm),
                ("Episode", f"{getattr(self, 'episode', 0)}"),
                ("Total Steps", f"{getattr(self, 'total_steps', 0)}"),
                ("Episode Steps", f"{self.steps}"),
                ("Episode Reward", f"{getattr(self, 'episode_reward', 0):.2f}"),
                ("Lane Changes", f"{getattr(self, 'lane_changes', 0)}")
            ]
            
            for param, value in training_info:
                text = self.font.render(f"{param}: {value}", True, (0, 0, 0))
                self.screen.blit(text, (info_x, info_y))
                info_y += line_height
            
            info_y += line_height
            
            # 3. 实时状态 (Real-time Status)
            title = self.font.render("Real-time Status", True, (0, 0, 0))
            self.screen.blit(title, (info_x, info_y))
            info_y += line_height * 1.5
            
            current_info = [
                ("Current Speed", f"{self.controlled_vehicle.velocity[0] * 3.6:.1f} km/h"),
                ("Current Lane", f"{self.controlled_vehicle.lane_id + 1}"),
                ("Current Reward", f"{self._compute_reward():.2f}"),
                ("Simulation Time", f"{self.time:.1f} s")
            ]
            
            for param, value in current_info:
                text = self.font.render(f"{param}: {value}", True, (0, 0, 0))
                self.screen.blit(text, (info_x, info_y))
                info_y += line_height
            
            # 4. 碰撞警告 (Collision Warning)
            for vehicle in self.vehicles:
                if vehicle != self.controlled_vehicle:
                    distance = np.linalg.norm(
                        vehicle.position - self.controlled_vehicle.position
                    )
                    if distance < (vehicle.length + self.controlled_vehicle.length):
                        warning = self.font.render("WARNING: Collision Risk!", True, (255, 0, 0))
                        self.screen.blit(warning, (info_x, info_y))
                        break
            
            if self.render_mode == "human":
                try:
                    pygame.display.flip()
                    self.clock.tick(self.metadata["render_fps"])
                    return None
                except pygame.error:
                    print("Warning: Pygame display error")
                    self.close()
                    return None
            
            return self.screen
            
        except Exception as e:
            print(f"Render error: {e}")
            self.close()
            return None
    
    def _compute_reward(self) -> float:
        """计算奖励 (表2-5)"""
        reward = 0.0
        
        # 速度奖励
        speed = self.controlled_vehicle.velocity[0]
        speed_reward = 1/12 * np.clip(speed * 3.6, -2, 10)  # 转换为km/h
        reward += speed_reward
        
        # 碰撞惩罚
        for vehicle in self.vehicles:
            if vehicle != self.controlled_vehicle:
                # 计算距离
                distance = np.linalg.norm(
                    vehicle.position - self.controlled_vehicle.position
                )
                if distance < (vehicle.length + self.controlled_vehicle.length) / 2:
                    speed = self.controlled_vehicle.velocity[0]
                    reward += -10 + speed/12  # 碰撞惩罚
                    return reward
        
        # 时间惩罚
        time_penalty = -self.time/400
        reward += time_penalty
        
        # 完成奖励
        if self.time >= self.config.simulation.duration:
            reward += 5
        
        # 动作惩罚
        reward += 1
        
        return reward
    
    def _is_terminated(self) -> bool:
        """检是否终止"""
        # 检查碰撞
        for vehicle in self.vehicles:
            if vehicle != self.controlled_vehicle:
                distance = np.linalg.norm(
                    vehicle.position - self.controlled_vehicle.position
                )
                if distance < (vehicle.length + self.controlled_vehicle.length) / 2:
                    return True
        return False
    
    def _update_state(self, action):
        """更新状态"""
        # 这里需要实现状态更新逻辑
        pass
    
    def _init_other_vehicles(self):
        """初始化其他车辆"""
        # 这里需要实现其他车辆的初始化
        pass
    
    def close(self):
        """关闭环境"""
        try:
            if self.screen is not None:
                import pygame
                pygame.display.quit()
                pygame.quit()
        except Exception as e:
            print(f"Close error: {e}")
        finally:
            self.screen = None
            self.clock = None