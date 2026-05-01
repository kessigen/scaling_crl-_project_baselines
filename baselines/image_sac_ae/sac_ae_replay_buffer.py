
import pickle
from pathlib import Path

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, obs_shape, action_shape, capacity, device):
        self.capacity = capacity
        self.device = torch.device(device)

        self.obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=np.uint8)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones_no_max = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False

    def __len__(self):
        return self.capacity if self.full else self.idx

    def add(
        self,
        obs,
        action,
        reward,
        next_obs,
        done,
        done_no_max,
    ):
        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], 0.0 if done else 1.0)
        np.copyto(self.not_dones_no_max[self.idx], 0.0 if done_no_max else 1.0)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample(self, batch_size):
        idxs = np.random.randint(0, self.capacity if self.full else self.idx, size=batch_size)
        obses = torch.as_tensor(self.obses[idxs], device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(self.next_obses[idxs], device=self.device).float()
        not_dones_no_max = torch.as_tensor(self.not_dones_no_max[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones_no_max

    def _latest_indices(self, max_entries = None):
        size = self.capacity if self.full else self.idx
        if size == 0:
            return np.empty((0,), dtype=np.int64)

        if max_entries is None or max_entries >= size:
            if self.full:
                return np.concatenate(
                    [
                        np.arange(self.idx, self.capacity, dtype=np.int64),
                        np.arange(0, self.idx, dtype=np.int64),
                    ]
                )
            return np.arange(0, size, dtype=np.int64)

        keep = max(1, int(max_entries))
        if self.full:
            start = (self.idx - keep) % self.capacity
            if start < self.idx:
                return np.arange(start, self.idx, dtype=np.int64)
            return np.concatenate(
                [
                    np.arange(start, self.capacity, dtype=np.int64),
                    np.arange(0, self.idx, dtype=np.int64),
                ]
            )

        start = max(0, size - keep)
        return np.arange(start, size, dtype=np.int64)

    def state_dict(self, max_entries = None):
        indices = self._latest_indices(max_entries)
        size = len(indices)
        return {
            "obses": self.obses[indices],
            "next_obses": self.next_obses[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "not_dones": self.not_dones[indices],
            "not_dones_no_max": self.not_dones_no_max[indices],
            "idx": size % self.capacity if self.capacity > 0 else 0,
            "full": size == self.capacity and self.capacity > 0,
            "capacity": self.capacity,
        }

    def load_state_dict(self, state):
        size = len(state["actions"])
        self.obses[:size] = state["obses"]
        self.next_obses[:size] = state["next_obses"]
        self.actions[:size] = state["actions"]
        self.rewards[:size] = state["rewards"]
        self.not_dones[:size] = state["not_dones"]
        self.not_dones_no_max[:size] = state["not_dones_no_max"]
        self.idx = state["idx"]
        self.full = state["full"]

    def save(self, path, max_entries = None):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self.state_dict(max_entries=max_entries), handle)

    def load(self, path):
        with path.open("rb") as handle:
            state = pickle.load(handle)
        self.load_state_dict(state)
