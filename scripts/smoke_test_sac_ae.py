
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import torch

from baselines.image_sac_ae.chw_wrapper import ImageToCHW
from baselines.image_sac_ae.sac_ae_agent import SACAEAgent
from baselines.image_sac_ae.sac_ae_replay_buffer import ReplayBuffer
from envs.wx250_pick_env_image import WX250PickPlaceImageEnv


def main():
    env = ImageToCHW(
        WX250PickPlaceImageEnv(
            obs_mode="image",
            obs_height=84,
            obs_width=84,
            domain_randomize=False,
        )
    )
    obs, _ = env.reset(seed=0)
    assert obs.shape == (3, 84, 84), obs.shape
    assert obs.dtype == "uint8" or str(obs.dtype) == "uint8"

    device = "cuda" if torch.cuda.is_available() else "cpu"
    agent = SACAEAgent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_range=(float(env.action_space.low.min()), float(env.action_space.high.max())),
        device=device,
        feature_dim=50,
        hidden_dim=256,
    )
    replay = ReplayBuffer(env.observation_space.shape, env.action_space.shape, capacity=2048, device=device)

    obs, _ = env.reset(seed=0)
    for step in range(150):
        if step < 20:
            action = env.action_space.sample()
        else:
            action = agent.act(obs, sample=True)
        next_obs, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        replay.add(obs, action, float(reward), next_obs, done, bool(terminated))
        obs = next_obs
        if done:
            obs, _ = env.reset()
        if len(replay) >= 64:
            metrics = agent.update(replay, batch_size=64, step=step + 1)
            assert metrics.recon_loss >= 0.0

    print("[ok] SAC+AE smoke test completed")
    env.close()
    raise SystemExit(0)


if __name__ == "__main__":
    main()
