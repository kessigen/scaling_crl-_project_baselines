
from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces


class DrQFrameStack(gym.Wrapper):
    """Stack HWC images into a CHW tensor as expected by the original DrQ code."""

    def __init__(self, env, k):
        super().__init__(env)
        self.k = k
        self.frames = deque([], maxlen=k)

        assert isinstance(env.observation_space, spaces.Box)
        h, w, c = env.observation_space.shape
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(c * k, h, w),
            dtype=np.uint8,
        )

    def _transform(self, obs):
        chw = np.transpose(obs, (2, 0, 1))
        return chw.copy()

    def _get_obs(self):
        assert len(self.frames) == self.k
        return np.concatenate(list(self.frames), axis=0)

    def reset(self, *, seed = None, options = None):
        obs, info = self.env.reset(seed=seed, options=options)
        frame = self._transform(obs)
        for _ in range(self.k):
            self.frames.append(frame)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(self._transform(obs))
        return self._get_obs(), reward, terminated, truncated, info
