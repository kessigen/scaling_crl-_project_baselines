
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack


from baselines.image_sac_pixel_encoder.custom_pixel_encoder import PixelEncoder  # noqa: E402
from envs.wx250_pick_env_image import WX250PickPlaceImageEnv  # noqa: E402


def main():
    venv = VecFrameStack(
        DummyVecEnv([lambda: WX250PickPlaceImageEnv(obs_mode="image", domain_randomize=False)]),
        n_stack=3,
    )
    model = SAC(
        "CnnPolicy",
        venv,
        buffer_size=2_000,
        learning_starts=50,
        batch_size=32,
        optimize_memory_usage=True,
        replay_buffer_kwargs=dict(handle_timeout_termination=False),
        policy_kwargs=dict(
            features_extractor_class=PixelEncoder,
            features_extractor_kwargs=dict(features_dim=50),
            net_arch=[128, 128],
            normalize_images=True,
        ),
        verbose=0,
    )
    model.learn(total_timesteps=100)
    print("[ok] SAC CnnPolicy + custom pixel encoder completed 100 training timesteps")
    venv.close()


if __name__ == "__main__":
    main()
