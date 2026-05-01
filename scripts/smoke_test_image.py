"""Smoke test for the image-based env: verify obs shape, save rendered frames, and confirm VecFrameStack."""

import os
import sys as _sys
# On Linux/Colab we want EGL for headless GPU rendering; Windows has no EGL, use its native GL.
if _sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
from PIL import Image

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from envs.wx250_pick_env_image import WX250PickPlaceImageEnv


def main():
    out_dir = HERE / "smoke_frames"
    out_dir.mkdir(exist_ok=True)

    # 1) Single env: basic obs/action checks.
    env = WX250PickPlaceImageEnv(obs_mode="image", obs_height=84, obs_width=84, domain_randomize=True)
    obs, info = env.reset(seed=0)
    assert obs.dtype == np.uint8, f"expected uint8, got {obs.dtype}"
    assert obs.shape == (84, 84, 3), f"expected (84,84,3), got {obs.shape}"
    print(f"[ok] obs shape={obs.shape} dtype={obs.dtype}")
    print(f"[ok] observation_space={env.observation_space}")
    print(f"[ok] action_space={env.action_space}")

    Image.fromarray(obs).save(out_dir / "obs_reset_seed0.png")

    # Save a few randomized resets so we can eyeball domain randomization.
    for i in range(4):
        o, _ = env.reset(seed=100 + i)
        Image.fromarray(o).save(out_dir / f"obs_reset_seed{100 + i}.png")

    # Step a few times with random actions.
    last_reward = None
    for t in range(5):
        o, r, term, trunc, info = env.step(env.action_space.sample())
        last_reward = r
        print(f"step {t}: reward={r:+.3f}  term={term}  trunc={trunc}  success={info['is_success']}")
    Image.fromarray(o).save(out_dir / "obs_after_steps.png")
    assert last_reward is not None
    env.close()

    # 2) Pose mode still works for sanity (lets us share env across baselines).
    env_pose = WX250PickPlaceImageEnv(obs_mode="pose", domain_randomize=False)
    pobs, _ = env_pose.reset(seed=0)
    print(f"[ok] pose obs shape={pobs.shape} dtype={pobs.dtype}")
    env_pose.close()

    # 3) VecFrameStack pipeline as used in training.
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    venv = VecFrameStack(DummyVecEnv([lambda: WX250PickPlaceImageEnv(obs_mode="image")]), n_stack=3)
    stacked = venv.reset()
    print(f"[ok] VecFrameStack obs shape={stacked.shape} dtype={stacked.dtype}")
    assert stacked.shape == (1, 84, 84, 9), f"expected (1,84,84,9), got {stacked.shape}"
    venv.close()

    # 4) One SAC training step to catch CnnPolicy wiring errors early.
    from stable_baselines3 import SAC
    venv2 = VecFrameStack(DummyVecEnv([lambda: WX250PickPlaceImageEnv(obs_mode="image", domain_randomize=False)]), n_stack=3)
    model = SAC(
        "CnnPolicy",
        venv2,
        buffer_size=2_000,
        learning_starts=50,
        batch_size=32,
        optimize_memory_usage=True,
        replay_buffer_kwargs=dict(handle_timeout_termination=False),
        policy_kwargs=dict(net_arch=[64, 64], normalize_images=True),
        verbose=0,
    )
    model.learn(total_timesteps=100)
    print("[ok] SAC CnnPolicy completed 100 training timesteps")
    venv2.close()

    print(f"\nSmoke frames written to: {out_dir}")


if __name__ == "__main__":
    main()
