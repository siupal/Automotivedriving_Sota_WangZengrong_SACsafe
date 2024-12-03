import gymnasium as gym
import highway_env

env = gym.make('Highway-v0', render_mode='human')
observation, info = env.reset()

try:
    for _ in range(1000):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        
        if terminated or truncated:
            observation, info = env.reset()
finally:
    env.close()  # 确保正确关闭环境