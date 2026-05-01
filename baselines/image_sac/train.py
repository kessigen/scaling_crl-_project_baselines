
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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
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

from envs.wx250_pick_env_image import WX250PickPlaceImageEnv  # noqa: E402


class PruningCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that deletes older replay-buffer .pkl files after a new save.

    Replay buffers are tens of GB; we only ever resume from the most recent one,
    so older pkls just waste disk. Model .zip checkpoints are kept untouched.
    """

    def _on_step(self):
        will_save = self.n_calls % self.save_freq == 0
        result = super()._on_step()
        if will_save and self.save_replay_buffer:
            save_dir = Path(self.save_path)
            current = save_dir / f"{self.name_prefix}_replay_buffer_{self.num_timesteps}_steps.pkl"
            for pkl in save_dir.glob(f"{self.name_prefix}_replay_buffer_*_steps.pkl"):
                if pkl.resolve() != current.resolve():
                    try:
                        pkl.unlink()
                    except OSError as e:
                        print(f"[PruningCheckpointCallback] failed to delete {pkl}: {e}")
        return result


class WandbMetricsCallback(BaseCallback):
    def __init__(self, log_freq = 1000):
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

    def _on_step(self):
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
    def _on_step(self):
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


def make_env(
    asset_root,
    render_mode=None,
    seed = 0,
    obs_height = 64,
    obs_width = 64,
    domain_randomize = True,
    obs_camera = "front",
    easy_reset_prob = 0.0,
    easy_distance_scale = 0.5,
):
    def _thunk():
        env = WX250PickPlaceImageEnv(
            render_mode=render_mode,
            asset_root=asset_root,
            obs_mode="image",
            obs_height=obs_height,
            obs_width=obs_width,
            domain_randomize=domain_randomize,
            obs_camera=obs_camera,
            seed=seed,
            easy_reset_prob=easy_reset_prob,
            easy_distance_scale=easy_distance_scale,
        )
        return Monitor(env)
    return _thunk


def replay_buffer_path_from_checkpoint(checkpoint_path):
    checkpoint_path = Path(checkpoint_path)
    match = re.match(r"(?P<prefix>.+)_(?P<steps>\d+)_steps$", checkpoint_path.stem)
    if match is None:
        return None
    return checkpoint_path.with_name(
        f"{match.group('prefix')}_replay_buffer_{match.group('steps')}_steps.pkl"
    )


def main():
    parser = argparse.ArgumentParser()
    default_asset_root = str(
        REPO_ROOT
        / "assets" / "external" / "interbotix_ros_manipulators" / "interbotix_ros_xsarms"
        / "interbotix_xsarm_descriptions" / "meshes"
    )
    parser.add_argument("--asset-root", default=default_asset_root)
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--run-dir", default="runs/wx250_image_sac")
    parser.add_argument(
        "--drive-dir",
        default=None,
        help="Optional Google Drive root for persistent Colab outputs, e.g. /content/drive/MyDrive/wx250_runs",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    parser.add_argument("--n-envs", type=int, default=1, help="Number of parallel rollout envs (1 = DummyVecEnv, >1 = SubprocVecEnv)")
    parser.add_argument("--obs-size", type=int, default=64, help="Square observation resolution")
    parser.add_argument("--obs-camera", choices=WX250PickPlaceImageEnv.OBS_CAMERAS, default="front", help="Observation camera to render for image observations.")
    parser.add_argument("--n-stack", type=int, default=3, help="Frames to stack via VecFrameStack")
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--no-randomize", action="store_true", help="Disable domain randomization (for debugging)")
    parser.add_argument(
        "--easy-reset-prob",
        type=float,
        default=0.0,
        help="Probability of using an easier reset that shrinks cube/goal distances during training only.",
    )
    parser.add_argument(
        "--easy-distance-scale",
        type=float,
        default=0.5,
        help="Distance shrink factor for easier resets (0 < scale <= 1, smaller is easier).",
    )
    parser.add_argument("--wandb-dir", default=None, help="Local dir for wandb logs (defaults to /tmp/wandb). Keep OFF Drive - Drive FUSE stalls training.")
    parser.add_argument(
        "--checkpoint-freq",
        type=int,
        default=50_000,
        help="Checkpoint frequency in total env timesteps. Internally scaled by n-envs for SB3 callbacks.",
    )
    parser.add_argument("--no-checkpoints", action="store_true", help="Disable periodic checkpointing. Only the final model and replay buffer are saved at end of training.")
    parser.add_argument("--no-replay-buffer", action="store_true", help="Never save the replay buffer (.pkl) - skips both periodic and final buffer saves. Resumes via --resume will refill buffer from scratch.")
    args = parser.parse_args()

    storage_root = Path(args.drive_dir) if args.drive_dir is not None else Path(".")
    storage_root.mkdir(parents=True, exist_ok=True)
    run_dir = storage_root / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    # Keep wandb logs on fast local disk - writing through Google Drive FUSE
    # stalls the training loop (~50-200 ms per event). Drive is for checkpoints only.
    wandb_dir = Path(args.wandb_dir) if args.wandb_dir else Path("/tmp/wandb")
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
        ckpts = glob.glob(str(run_dir / "checkpoints" / "wx250_image_sac_*_steps.zip"))
        def _ckpt_step(p):
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
    env = None
    eval_env = None
    model = None

    try:
        env_fns = [
            make_env(
                args.asset_root,
                seed=i,
                obs_height=args.obs_size,
                obs_width=args.obs_size,
                domain_randomize=domain_randomize,
                obs_camera=args.obs_camera,
                easy_reset_prob=args.easy_reset_prob,
                easy_distance_scale=args.easy_distance_scale,
            )
            for i in range(n_envs)
        ]
        eval_env_fns = [
            make_env(
                args.asset_root,
                seed=1000 + i,
                obs_height=args.obs_size,
                obs_width=args.obs_size,
                domain_randomize=domain_randomize,
                obs_camera=args.obs_camera,
            )
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

        wandb_init_kwargs = {
            "project": "wx250-pick-place",
            "config": {
                "algorithm": "SAC",
                "policy": "CnnPolicy",
                "obs_mode": "image",
                "obs_size": args.obs_size,
                "obs_camera": args.obs_camera,
                "n_stack": args.n_stack,
                "n_envs": n_envs,
                "domain_randomize": domain_randomize,
                "easy_reset_prob": args.easy_reset_prob,
                "easy_distance_scale": args.easy_distance_scale,
                "net_arch": [256, 256],
                "learning_rate": args.learning_rate,
                "buffer_size": args.buffer_size,
                "learning_starts": args.learning_starts,
                "batch_size": args.batch_size,
                "tau": 0.005,
                "gamma": 0.99,
                "gradient_steps": gradient_steps,
                "checkpoint_freq": args.checkpoint_freq,
                "total_timesteps": args.timesteps,
            },
            "sync_tensorboard": False,
            "dir": str(wandb_dir.resolve()),
            "settings": wandb.Settings(init_timeout=300),
        }
        wandb.init(**wandb_init_kwargs)
        wandb_id_file.write_text(wandb.run.id)

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

        callbacks = [eval_cb, wandb_cb]
        if not args.no_checkpoints:
            checkpoint_cb = PruningCheckpointCallback(
                save_freq=max(args.checkpoint_freq // n_envs, 1),
                save_path=str(run_dir / "checkpoints"),
                name_prefix="wx250_image_sac",
                save_replay_buffer=not args.no_replay_buffer,
            )
            callbacks.insert(0, checkpoint_cb)

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
                ent_coef="auto_0.1",
                optimize_memory_usage=True,
                replay_buffer_kwargs=dict(handle_timeout_termination=False),
                verbose=1,
            )

        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=checkpoint_path is None,
        )
        model.save(str(run_dir / "final_model"))
        if not args.no_replay_buffer:
            model.save_replay_buffer(str(run_dir / "final_replay_buffer"))
        else:
            print("Skipping final_replay_buffer.pkl save (--no-replay-buffer).")
    finally:
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
