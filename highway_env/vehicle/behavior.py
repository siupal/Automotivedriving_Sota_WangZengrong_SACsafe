import numpy as np
from typing import Dict, Optional, Tuple

class IDM:
    """Intelligent Driver Model (IDM) 实现"""
    def __init__(self, params: Dict):
        """
        初始化IDM模型参数 (表2-1)
        
        Args:
            params: IDM参数字典
                - a_max: 最大加速度 ω = 5.0 m/s²
                - b_comfort: 最大减速度 b = -4.0 m/s²
                - time_headway: 安全时间间隔 T = 1.5 s
                - distance_min: 最小相对距离 d₀ = 10.0 m
                - delta: 加速度指数 δ = 4.0
        """
        self.a_max = params["a_max"]  # ω
        self.b_comfort = abs(params["b_comfort"])  # b (使用绝对值)
        self.time_headway = params["time_headway"]  # T
        self.distance_min = params["distance_min"]  # d₀
        self.delta = params["delta"]  # δ
        
    def compute_acceleration(self, v: float, delta_v: float, distance: float) -> float:
        """
        计算IDM加速度 (公式2-1和2-2)
        
        Args:
            v: 当前速度(m/s)
            delta_v: 相对速度(m/s)
            distance: 车间距离(m)
            
        Returns:
            加速度(m/s²)
        """
        # 如果距离太小，紧急制动
        if distance <= 0:
            return -self.b_comfort
            
        # 计算期望距离 (公式2-2)
        d_star = (
            self.distance_min +  # 最小距离
            v * self.time_headway +  # 速度时距项
            max(0, v * delta_v / (2 * np.sqrt(self.a_max * self.b_comfort)))  # 智能制动项
        )
        
        # 计算加速度 (公式2-1)
        v0 = 120/3.6  # 目标速度
        a = self.a_max * (
            1 - (v/v0)**self.delta -  # 自由加速项
            (d_star/distance)**2  # 减速项
        )
        
        return np.clip(a, -self.b_comfort, self.a_max)  # 限制加速度范围

class MOBIL:
    """MOBIL换道模型"""
    
    def __init__(self, params: Optional[Dict] = None):
        if params is None:
            params = {}
            
        # 基本参数 (调整为更激进的换道策略)
        self.politeness = params.get("politeness", 0.3)  # 降低礼让因子
        self.safe_threshold = params.get("safe_threshold", -2.0)  # 提高安全阈值容忍度
        self.acceleration_threshold = params.get("acceleration_threshold", 0.05)  # 降低换道阈值
        
        # 随机性和倾向性参数
        self.random_threshold = params.get("random_threshold", 0.3)  # 增加随机性
        self.bias_left = params.get("bias_left", 0.2)  # 增加左侧超车倾向
        self.overtake_speed_threshold = params.get("overtake_speed_threshold", 5.0)  # 超车速度阈值(m/s)
        self.last_decision_time = 0.0
        
    def can_change_lane(self, ego_vehicle, new_following, new_leading, old_following=None, old_leading=None) -> Tuple[bool, float]:
        """判断是否可以换道"""
        # 添加随机性和超车倾向
        random_factor = np.random.uniform(-self.random_threshold, self.random_threshold)
        
        # 安全性检查 (稍微放宽限制)
        if new_following and new_following.idm:
            acc = new_following.idm.compute_acceleration(
                new_following.velocity[0],
                ego_vehicle.velocity[0] - new_following.velocity[0],
                ego_vehicle.position[0] - new_following.position[0]
            )
            if acc < self.safe_threshold:
                return False, 0.0
        
        advantage = 0.0
        
        # 主车收益 (增加超车动机)
        if ego_vehicle.idm:
            # 当前车道加速度
            old_acc = ego_vehicle.idm.compute_acceleration(
                ego_vehicle.velocity[0],
                old_leading.velocity[0] - ego_vehicle.velocity[0] if old_leading else 0,
                old_leading.position[0] - ego_vehicle.position[0] if old_leading else float('inf')
            )
            
            # 目标车道加速度
            new_acc = ego_vehicle.idm.compute_acceleration(
                ego_vehicle.velocity[0],
                new_leading.velocity[0] - ego_vehicle.velocity[0] if new_leading else 0,
                new_leading.position[0] - ego_vehicle.position[0] if new_leading else float('inf')
            )
            
            # 基础优势
            advantage += (new_acc - old_acc) * (1 + random_factor)
            
            # 超车动机：如果前车速度慢，增加换道意愿
            if old_leading and ego_vehicle.velocity[0] - old_leading.velocity[0] > self.overtake_speed_threshold:
                advantage += 0.5  # 增加超车奖励
        
        # 左侧换道偏好 (增加超车倾向)
        if ego_vehicle.lane_id < 3:  # 如果不在最左侧
            advantage += self.bias_left * (1 + random_factor)  # 增加随机性
            
            # 如果左侧车道更快，增加换道意愿
            if new_leading and old_leading:
                if new_leading.velocity[0] > old_leading.velocity[0]:
                    advantage += 0.3  # 奖励换到更快的车道
        
        # 考虑其他车辆的影响 (降低礼让度)
        if new_following and new_following.idm:
            advantage += self.politeness * (
                new_following.idm.compute_acceleration(
                    new_following.velocity[0],
                    ego_vehicle.velocity[0] - new_following.velocity[0],
                    ego_vehicle.position[0] - new_following.position[0]
                ) - new_following.idm.compute_acceleration(
                    new_following.velocity[0],
                    new_leading.velocity[0] - new_following.velocity[0] if new_leading else 0,
                    new_leading.position[0] - new_following.position[0] if new_leading else float('inf')
                )
            ) * (1 + random_factor)
        
        # 降低换道阈值，增加换道频率
        threshold = self.acceleration_threshold * (0.7 + 0.3 * np.random.random())
        
        return advantage > threshold, advantage