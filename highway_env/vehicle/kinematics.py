from typing import Tuple, Optional, List, Dict
import numpy as np
from highway_env.vehicle.behavior import IDM, MOBIL

class Vehicle:
    """车辆类，包含运动学模型和行为模型"""
    
    def __init__(self, config: Dict):
        """初始化车辆"""
        # 基本属性
        self.length = config["length"]
        self.width = config["width"]
        self.controlled = config["controlled"]
        
        # 状态
        self.position = np.array(config["position"], dtype=np.float64)
        self.velocity = np.array([config["initial_speed"], 0], dtype=np.float64)
        self.heading = 0.0
        self.lane_id = config["lane_id"]
        
        # 行为模型
        if not self.controlled:
            # NPC车辆必须有IDM和MOBIL模型
            if "idm_params" not in config or "mobil_params" not in config:
                raise ValueError("NPC车辆需要IDM和MOBIL参数")
            self.idm = IDM(config["idm_params"])
            self.mobil = MOBIL(config["mobil_params"])
            self.lane_change_cooldown = 0.0
        else:
            # 主车不要行为模型
            self.idm = None
            self.mobil = None
            self.lane_change_cooldown = 0.0
        
        # 记录上一次的车道ID，用于检测换道
        self.last_lane_id = self.lane_id
    
    def step(self, dt: float, action: Optional[np.ndarray] = None, env=None):
        """更新车辆状态"""
        if self.controlled:
            # 主车：使用action控制
            if action is not None:
                acceleration, steering = action
                self._apply_control(acceleration, steering, dt)
        else:
            # NPC车辆：使用IDM和MOBIL模型
            if self.idm and self.mobil:
                # 1. 获取周围车辆
                front_vehicle, rear_vehicle = self._get_surrounding_vehicles(env)
                
                # 2. IDM纵向控制
                acceleration = self._idm_control(front_vehicle)
                
                # 3. MOBIL换道决策
                if self.lane_change_cooldown <= 0:
                    lane_changed = self._mobil_decision(env)
                    if lane_changed:
                        self.lane_change_cooldown = 2.0
                else:
                    self.lane_change_cooldown -= dt
                
                # 4. 应用控制
                self._apply_control(acceleration, 0, dt)
            else:
                # 如果没有行为模型，保持匀速运动
                self._apply_control(0, 0, dt)
        
        # 确保车辆在道路范围内
        if env is not None:
            self._clip_position(env)
    
    def _apply_control(self, acceleration: float, steering: float, dt: float):
        """应用控制输入"""
        # 更新速度 (表2-4限速120km/h)
        self.velocity[0] = np.clip(
            self.velocity[0] + acceleration * dt,
            0,
            120/3.6  # 最大速度120km/h
        )
        
        # 更新航向角
        self.heading = np.clip(
            self.heading + steering * dt,
            -np.pi/4,  # 最大转向角
            np.pi/4
        )
        
        # 更新速度向量
        speed = np.linalg.norm(self.velocity)
        self.velocity = speed * np.array([
            np.cos(self.heading),
            np.sin(self.heading)
        ])
        
        # 更新位置
        self.position += self.velocity * dt
        
        # 确保车辆在车道内
        self.position[1] = np.clip(
            self.position[1],
            0,  # 最右车道
            (self.lane_id + 1) * 3.75  # 当前车道中心线
        )
    
    def _get_surrounding_vehicles(self, env) -> Tuple[Optional['Vehicle'], Optional['Vehicle']]:
        """获取前后车辆"""
        front_vehicle = None
        rear_vehicle = None
        min_front_distance = float('inf')
        min_rear_distance = float('inf')
        
        for vehicle in env.vehicles:
            if vehicle != self and vehicle.lane_id == self.lane_id:
                # 计算纵向距离
                dx = vehicle.position[0] - self.position[0]
                if dx > 0 and dx < min_front_distance:  # 前车
                    front_vehicle = vehicle
                    min_front_distance = dx
                elif dx < 0 and -dx < min_rear_distance:  # 后车
                    rear_vehicle = vehicle
                    min_rear_distance = -dx
                    
        return front_vehicle, rear_vehicle
    
    def _idm_control(self, front_vehicle=None) -> float:
        """使用IDM模型计算加速度"""
        if not hasattr(self, 'idm') or front_vehicle is None:
            # 如果没有IDM模型或前车，使用默认加速度
            return 0.0
        
        # 计算相对速度和距离
        delta_x = front_vehicle.position[0] - self.position[0]  # 相对距离
        delta_v = self.velocity[0] - front_vehicle.velocity[0]  # 相对速度
        
        # 使用IDM模型计算加速度
        return self.idm.compute_acceleration(
            v=self.velocity[0],
            delta_v=delta_v,
            distance=max(delta_x, 1e-5)
        )
    
    def _mobil_decision(self, env):
        """MOBIL换道决策"""
        # 降低跳过换道的概率
        if np.random.random() < 0.05:  # 只有5%概率跳过换道决策
            return False
        
        # 获取左右车道的车辆
        left_vehicles = self._get_lane_vehicles(env, self.lane_id + 1)
        right_vehicles = self._get_lane_vehicles(env, self.lane_id - 1)
        
        # 优先考虑左侧换道 (增加概率)
        if self.lane_id < env.config.road.lanes - 1:
            can_change_left, left_advantage = self.mobil.can_change_lane(
                self, *left_vehicles
            )
            if can_change_left:
                if np.random.random() < 0.9:  # 90%概率立即换道
                    self.lane_id += 1
                    self.position[1] = env.config.road.lane_centers[self.lane_id]
                    return True
        
        # 其次考虑右侧换道 (也增加概率)
        if self.lane_id > 0:
            can_change_right, right_advantage = self.mobil.can_change_lane(
                self, *right_vehicles
            )
            if can_change_right:
                if np.random.random() < 0.8:  # 80%概率立即换道
                    self.lane_id -= 1
                    self.position[1] = env.config.road.lane_centers[self.lane_id]
                    return True
        
        return False
    
    def _get_lane_vehicles(self, env, lane_id: int) -> Tuple[Optional['Vehicle'], Optional['Vehicle'], Optional['Vehicle']]:
        """获取指定车道的相关车辆（前车，后车）"""
        if lane_id < 0 or lane_id >= env.config.road.lanes:
            return None, None, None
            
        front_vehicle = None
        rear_vehicle = None
        min_front_distance = float('inf')
        min_rear_distance = float('inf')
        
        for vehicle in env.vehicles:
            if vehicle != self and vehicle.lane_id == lane_id:
                dx = vehicle.position[0] - self.position[0]
                if dx > 0 and dx < min_front_distance:
                    front_vehicle = vehicle
                    min_front_distance = dx
                elif dx < 0 and -dx < min_rear_distance:
                    rear_vehicle = vehicle
                    min_rear_distance = -dx
                    
        return front_vehicle, rear_vehicle, None
    
    def _clip_position(self, env):
        """确保车辆在道路范围内"""
        # 横向限制
        lane_width = env.lane_width if hasattr(env, 'lane_width') else 3.75
        n_lanes = env.lanes if hasattr(env, 'lanes') else 4
        
        self.position[1] = np.clip(
            self.position[1],
            0,
            (n_lanes - 1) * lane_width
        )
        
        # 纵向循环
        road_length = env.road_length if hasattr(env, 'road_length') else 1000
        if self.position[0] > road_length:
            self.position[0] = 0
        elif self.position[0] < 0:
            self.position[0] = road_length