from dataclasses import dataclass
import numpy as np
from typing import Dict, List, Tuple

@dataclass
class VehicleConfig:
    """车辆配置参数 (表2-4)"""
    length: float = 5.0  # 车辆长度(m)
    width: float = 1.8   # 车辆宽度(m)
    max_speed: float = 120/3.6  # 最大速度(m/s)
    initial_speed: float = 90/3.6  # 初始速度(m/s)
    
    # IDM模型参数 (表2-1)
    idm_params: Dict = None
    def __post_init__(self):
        if self.idm_params is None:
            self.idm_params = {
                "a_max": 5.0,  # 最大加速度(m/s²)
                "b_comfort": 4.0,  # 舒适减速度(m/s²)
                "time_headway": 1.5,  # 安全车头时距(s)
                "distance_min": 1.5,  # 最小安全距离(m)
                "delta": 4.0,  # 加速度指数
            }
    
    # MOBIL模型参数
    mobil_params: Dict = None
    def __post_init__(self):
        if self.mobil_params is None:
            self.mobil_params = {
                "politeness": 0.5,  # 礼让因子
                "safe_threshold": -4.0,  # 安全阈值(m/s²)
                "acceleration_threshold": 0.1,  # 换道阈值(m/s²)
                "lane_change_cooldown": 2.0,  # 换道冷却时间(s)
            }

@dataclass
class RoadConfig:
    """道路配置参数 (表2-4)"""
    lanes: int = 4  # 车道数
    lane_width: float = 3.75  # 车道宽度(m)
    length: float = 1000  # 道路长度(m)
    speed_limit: float = 120/3.6  # 限速(m/s)
    
    @property
    def lane_centers(self) -> List[float]:
        """车道中心线y坐标"""
        return [(i + 0.5) * self.lane_width for i in range(self.lanes)]

@dataclass
class TrafficConfig:
    """交通流配置"""
    n_vehicles: int = 8  # NPC车辆数量
    min_speed: float = 80/3.6  # 最低速度(m/s)
    max_speed: float = 120/3.6  # 最高速度(m/s)
    spawn_distance: float = 50  # 生成距离(m)
    spacing: float = 20  # 车辆间距(m)
    initial_lane_spread: bool = True  # 是否在不同车道上生成

@dataclass
class ObservationConfig:
    """观察空间配置 (图2-11)"""
    observation_range: float = 50.0  # 观察范围(m)
    n_observation_vehicles: int = 8  # 主车+7辆其他车辆
    
    observation_features: List[str] = None
    def __post_init__(self):
        self.observation_features = [
            'x', 'y',           # 位置
            'vx', 'vy',         # 速度
            'heading',          # 航向角
            'lane_id'           # 车道
        ]

@dataclass
class ActionConfig:
    """动作空间配置"""
    action_range: Dict = None
    def __post_init__(self):
        self.action_range = {
            'acceleration': [-5.0, 5.0],  # 加速度范围(m/s²)
            'steering': [-np.pi/4, np.pi/4]  # 转向角范围(rad)
        }

@dataclass
class RewardConfig:
    """奖励函数配置 (表2-5)"""
    weights: Dict = None
    def __post_init__(self):
        self.weights = {
            'collision': lambda v: -10 + v/12 if v else 0,  # 碰撞惩罚
            'speed': lambda v: 1/12 * np.clip(v, -2, 10),  # 速度奖励
            'time': lambda t: -t/400,  # 时间惩罚
            'success': 5,  # 完成奖励
            'action': 1  # 动作惩罚
        }

@dataclass
class SimulationConfig:
    """仿真配置参数 (表2-4)"""
    dt: float = 0.1  # 仿真步长(s)
    simulation_frequency: float = 10  # 仿真频率(Hz)
    duration: float = 40.0  # 场景持续时间(s)
    sampling_time: float = 1.0  # 采样时间(s)
    
    # 渲染参数
    screen_width: int = 800
    screen_height: int = 600
    scaling_factor: float = 10  # 像素/米
    
    # 随机参数
    random_seed: int = None

@dataclass
class EnvConfig:
    """环境总配置"""
    vehicle: VehicleConfig = VehicleConfig()
    road: RoadConfig = RoadConfig()
    traffic: TrafficConfig = TrafficConfig()
    observation: ObservationConfig = ObservationConfig()
    action: ActionConfig = ActionConfig()
    reward: RewardConfig = RewardConfig()
    simulation: SimulationConfig = SimulationConfig() 