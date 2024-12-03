from gymnasium.envs.registration import register

register(
    id='Highway-v0',
    entry_point='highway_env.envs.highway_env:HighwayEnv',
    max_episode_steps=1000,
)

from highway_env.envs.highway_env import HighwayEnv 