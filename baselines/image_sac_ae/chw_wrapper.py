
import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ImageToCHW(gym.Wrapper):
    """Convert image observations from HWC to CHW for PyTorch agents."""

    def __init__(self, env):
        super().__init__(env)
        assert isinstance(env.observation_space, spaces.Box)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(low=0, high=255, shape=(c, h, w), dtype=np.uint8)

    @staticmethod
    def _transform(obs):
        return np.transpose(obs, (2, 0, 1)).copy()

    def reset(self, *, seed = None, options = None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._transform(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self._transform(obs), reward, terminated, truncated, info
