from __future__ import annotations

import os
import sys as _sys

# On Linux/Colab headless we want EGL for GPU rendering; Windows has no EGL, use its native GL.
if _sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import argparse
import glob
import sys
import time
from pathlib import Path
import re

import numpy as np
import wandb

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

# Make the sibling env module importable whether we run as a module or script.
HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from wx250_pick_env_image import WX250PickPlaceImageEnv  # noqa: E402


class WandbMetricsCallback(BaseCallback):
    def __init__(self, log_freq: int = 1000):
        super().__init__()
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.total_successes = 0
        self.episode_final_reach = []
        self.episode_final_place = []
        self.reward_components = {"d_grip_cube": [], "d_cube_goal": [], "success": []}
        self._t0 = time.time()
        self._last_fps_step = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            bd = info.get("reward_breakdown", {})
            if bd:
                for key in self.reward_components:
                    self.reward_components[key].append(bd.get(key, 0.0))
            if dones[i]:
                success = info.get("is_success", False)
                self.episode_successes.append(float(success))
                if success:
                    self.total_successes += 1
                self.episode_final_reach.append(bd.get("d_grip_cube", 0.0))
                self.episode_final_place.append(bd.get("d_cube_goal", 0.0))
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])

        if self.num_timesteps % self.log_freq == 0:
            log = {}

            if self.episode_rewards:
                log["rollout/ep_reward_mean"] = float(np.mean(self.episode_rewards))
                log["rollout/ep_reward_min"] = float(np.min(self.episode_rewards))
                log["rollout/ep_reward_max"] = float(np.max(self.episode_rewards))
                log["rollout/ep_len_mean"] = float(np.mean(self.episode_lengths))
                self.episode_rewards.clear()
                self.episode_lengths.clear()

            if self.episode_successes:
                log["rollout/success_rate"] = float(np.mean(self.episode_successes))
                log["rollout/total_successes"] = self.total_successes
                self.episode_successes.clear()

            if self.episode_final_reach:
                log["rollout/final_reach_dist"] = float(np.mean(self.episode_final_reach))
                self.episode_final_reach.clear()

            if self.episode_final_place:
                log["rollout/final_place_dist"] = float(np.mean(self.episode_final_place))
                self.episode_final_place.clear()

            for key, vals in self.reward_components.items():
                if vals:
                    log[f"reward/{key}"] = float(np.mean(vals))
                    vals.clear()

            dt = time.time() - self._t0
            dstep = self.num_timesteps - self._last_fps_step
            if dt > 0 and dstep > 0:
                log["time/fps"] = dstep / dt
                log["time/total_seconds"] = dt
            self._t0 = time.time()
            self._last_fps_step = self.num_timesteps

            if self.model.logger is not None:
                logger = self.model.logger.name_to_value
                for key in ["train/actor_loss", "train/critic_loss", "train/ent_coef",
                            "train/ent_coef_loss", "train/learning_rate"]:
                    if key in logger:
                        log[key] = logger[key]

            wandb.log(log, step=self.num_timesteps)
        return True


class WandbEvalCallback(EvalCallback):
    def _on_step(self) -> bool:
        result = super()._on_step()
        if self.last_mean_reward is not None and self.num_timesteps % self.eval_freq == 0:
            log = {
                "eval/mean_reward": self.last_mean_reward,
                "eval/best_mean_reward": self.best_mean_reward,
            }
            if self.evaluations_results:
                last_rewards = np.asarray(self.evaluations_results[-1], dtype=np.float64)
                log["eval/std_reward"] = float(np.std(last_rewards))
            if self.evaluations_length:
                last_lengths = np.asarray(self.evaluations_length[-1], dtype=np.float64)
                log["eval/mean_ep_length"] = float(np.mean(last_lengths))
                log["eval/std_ep_length"] = float(np.std(last_lengths))
            if self._is_success_buffer:
                log["eval/success_rate"] = float(np.mean(self._is_success_buffer))
            wandb.log(log, step=self.num_timesteps)
        return result


class WandbImageCallback(BaseCallback):
    """Logs a periodic stack of sample observations to wandb so we can eyeball what the CNN sees."""

    def __init__(self, log_freq: int = 20_000, n_frames: int = 4):
        super().__init__()
        self.log_freq = log_freq
        self.n_frames = n_frames
        self._last_log = 0

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log < self.log_freq:
            return True
        self._last_log = self.num_timesteps
        # Sample a fresh env snapshot so we don't perturb the training rollout buffer.
        try:
            env = WX250PickPlaceImageEnv(domain_randomize=True)
            frames = []
            for i in range(self.n_frames):
                obs, _ = env.reset(seed=int(self.num_timesteps) + i)
                # obs is (H, W, 3) uint8
                frames.append(wandb.Image(obs, caption=f"reset {i}"))
            wandb.log({"samples/reset_frames": frames}, step=self.num_timesteps)
            env.close()
        except Exception as e:  # rendering failures shouldn't kill training
            print(f"[WandbImageCallback] skipped logging due to: {e}")
        return True


def make_env(asset_root: str, render_mode=None, seed: int = 0, obs_height: int = 84, obs_width: int = 84, domain_randomize: bool = True):
    def _thunk():
        env = WX250PickPlaceImageEnv(
            render_mode=render_mode,
            asset_root=asset_root,
            obs_mode="image",
            obs_height=obs_height,
            obs_width=obs_width,
            domain_randomize=domain_randomize,
            seed=seed,
        )
        return Monitor(env)
    return _thunk


def replay_buffer_path_from_checkpoint(checkpoint_path: str | Path) -> Path | None:
    checkpoint_path = Path(checkpoint_path)
    match = re.match(r"(?P<prefix>.+)_(?P<steps>\d+)_steps$", checkpoint_path.stem)
    if match is None:
        return None
    return checkpoint_path.with_name(
        f"{match.group('prefix')}_replay_buffer_{match.group('steps')}_steps.pkl"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    default_asset_root = str(
        Path(__file__).resolve().parent.parent
        / "external" / "interbotix_ros_manipulators" / "interbotix_ros_xsarms"
        / "interbotix_xsarm_descriptions" / "meshes"
    )
    parser.add_argument("--asset-root", default=default_asset_root)
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--run-dir", default="runs/wx250_sac_image")
    parser.add_argument(
        "--drive-dir",
        default=None,
        help="Optional Google Drive root for persistent Colab outputs, e.g. /content/drive/MyDrive/wx250_runs",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel rollout envs (1 = DummyVecEnv, >1 = SubprocVecEnv)")
    parser.add_argument("--obs-size", type=int, default=84, help="Square observation resolution (84 matches NatureCNN)")
    parser.add_argument("--n-stack", type=int, default=3, help="Frames to stack via VecFrameStack")
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--no-randomize", action="store_true", help="Disable domain randomization (for debugging)")
    parser.add_argument("--image-log-freq", type=int, default=20_000)
    args = parser.parse_args()

    storage_root = Path(args.drive_dir) if args.drive_dir is not None else Path(".")
    storage_root.mkdir(parents=True, exist_ok=True)
    run_dir = storage_root / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir = storage_root / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Set the WANDB_API_KEY environment variable before running.")
    os.environ["WANDB_DIR"] = str(wandb_dir.resolve())

    wandb_id_file = run_dir / "wandb_run_id.txt"
    wandb_id = None
    if args.resume and wandb_id_file.exists():
        wandb_id = wandb_id_file.read_text().strip()

    checkpoint_path = None
    resume_step = None
    if args.resume:
        ckpts = glob.glob(str(run_dir / "checkpoints" / "wx250_sac_image_*_steps.zip"))
        def _ckpt_step(p: str) -> int:
            m = re.search(r"_(\d+)_steps", Path(p).stem)
            return int(m.group(1)) if m else -1
        ckpts.sort(key=_ckpt_step)
        if ckpts:
            checkpoint_path = ckpts[-1]
        elif (run_dir / "final_model.zip").exists():
            checkpoint_path = str(run_dir / "final_model.zip")
        if checkpoint_path:
            print(f"Resuming from {checkpoint_path}")
            match = re.search(r"_(\d+)_steps", Path(checkpoint_path).stem)
            if match:
                resume_step = int(match.group(1))

    domain_randomize = not args.no_randomize
    n_envs = max(1, args.n_envs)
    VecCls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    env_fns = [
        make_env(args.asset_root, seed=i, obs_height=args.obs_size, obs_width=args.obs_size, domain_randomize=domain_randomize)
        for i in range(n_envs)
    ]
    eval_env_fns = [
        make_env(args.asset_root, seed=1000 + i, obs_height=args.obs_size, obs_width=args.obs_size, domain_randomize=domain_randomize)
        for i in range(1)
    ]
    # SubprocVecEnv on Windows/Colab needs start_method='spawn' for CUDA/OpenGL safety.
    if VecCls is SubprocVecEnv:
        env = SubprocVecEnv(env_fns, start_method="spawn")
    else:
        env = DummyVecEnv(env_fns)
    eval_env = DummyVecEnv(eval_env_fns)

    env = VecFrameStack(env, n_stack=args.n_stack)
    eval_env = VecFrameStack(eval_env, n_stack=args.n_stack)

    # Preserve SAC gradient-to-env-step ratio when running multiple envs: one grad step per env step total.
    gradient_steps = -1 if n_envs > 1 else 1

    wandb.init(
        project="wx250-pick-place",
        id=wandb_id,
        resume="allow" if wandb_id else None,
        config={
            "algorithm": "SAC",
            "policy": "CnnPolicy",
            "obs_mode": "image",
            "obs_size": args.obs_size,
            "n_stack": args.n_stack,
            "n_envs": n_envs,
            "domain_randomize": domain_randomize,
            "net_arch": [256, 256],
            "learning_rate": args.learning_rate,
            "buffer_size": args.buffer_size,
            "learning_starts": args.learning_starts,
            "batch_size": args.batch_size,
            "tau": 0.005,
            "gamma": 0.99,
            "gradient_steps": gradient_steps,
            "total_timesteps": args.timesteps,
        },
        sync_tensorboard=False,
        dir=str(wandb_dir.resolve()),
    )
    wandb_id_file.write_text(wandb.run.id)

    checkpoint_cb = CheckpointCallback(
        save_freq=max(50_000 // n_envs, 10_000),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="wx250_sac_image",
        save_replay_buffer=True,
    )
    eval_cb = WandbEvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=max(10_000 // n_envs, 2_500),
        deterministic=True,
        render=False,
        n_eval_episodes=10,
    )
    wandb_cb = WandbMetricsCallback(log_freq=max(1000, n_envs * 500))
    image_cb = WandbImageCallback(log_freq=args.image_log_freq, n_frames=4)

    if checkpoint_path:
        model = SAC.load(checkpoint_path, env=env)
        replay_buffer_path = replay_buffer_path_from_checkpoint(checkpoint_path)
        if replay_buffer_path is not None and replay_buffer_path.exists():
            model.load_replay_buffer(str(replay_buffer_path))
        else:
            print(f"Warning: replay buffer not found for checkpoint {checkpoint_path}; resuming weights only.")
    else:
        model = SAC(
            "CnnPolicy",
            env,
            policy_kwargs=dict(net_arch=[256, 256], normalize_images=True),
            learning_rate=args.learning_rate,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=gradient_steps,
            ent_coef="auto_0.01",
            optimize_memory_usage=True,
            replay_buffer_kwargs=dict(handle_timeout_termination=False),
            verbose=1,
        )

    model.learn(
        total_timesteps=args.timesteps,
        callback=[checkpoint_cb, eval_cb, wandb_cb, image_cb],
        progress_bar=True,
        reset_num_timesteps=checkpoint_path is None,
    )
    model.save(str(run_dir / "final_model"))
    model.save_replay_buffer(str(run_dir / "final_replay_buffer"))
    wandb.finish()


if __name__ == "__main__":
    main()
