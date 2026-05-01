
import argparse
import glob
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
import re

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import wandb

from stable_baselines3 import SAC
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.save_util import save_to_pkl
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecFrameStack

from baselines.image_sac_pixel_encoder.custom_pixel_encoder import PixelEncoder  # noqa: E402
from envs.wx250_pick_env_image import WX250PickPlaceImageEnv  # noqa: E402


class PruningCheckpointCallback(CheckpointCallback):
    def __init__(self, *args, saved_replay_buffer_size = None, **kwargs):
        kwargs["save_replay_buffer"] = False
        super().__init__(*args, **kwargs)
        self.saved_replay_buffer_size = saved_replay_buffer_size

    def _on_step(self):
        will_save = self.n_calls % self.save_freq == 0
        result = super()._on_step()
        if will_save and self.saved_replay_buffer_size is not None:
            save_dir = Path(self.save_path)
            current = save_dir / f"{self.name_prefix}_replay_buffer_{self.num_timesteps}_steps.pkl"
            save_replay_buffer_snapshot(self.model, current, self.saved_replay_buffer_size)
            for pkl in save_dir.glob(f"{self.name_prefix}_replay_buffer_*_steps.pkl"):
                if pkl.resolve() != current.resolve():
                    try:
                        pkl.unlink()
                    except OSError as exc:
                        print(f"[PruningCheckpointCallback] failed to delete {pkl}: {exc}")
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
                for key in [
                    "train/actor_loss",
                    "train/critic_loss",
                    "train/ent_coef",
                    "train/ent_coef_loss",
                    "train/learning_rate",
                ]:
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
    if checkpoint_path.stem == "final_model":
        return checkpoint_path.with_name("final_replay_buffer.pkl")
    match = re.match(r"(?P<prefix>.+)_(?P<steps>\d+)_steps$", checkpoint_path.stem)
    if match is None:
        return None
    return checkpoint_path.with_name(
        f"{match.group('prefix')}_replay_buffer_{match.group('steps')}_steps.pkl"
    )


def _latest_replay_indices(buffer, max_entries):
    size = buffer.buffer_size if buffer.full else buffer.pos
    if size == 0:
        return np.empty((0,), dtype=np.int64)

    if max_entries is None or max_entries >= size:
        if buffer.full:
            return np.concatenate(
                [
                    np.arange(buffer.pos, buffer.buffer_size, dtype=np.int64),
                    np.arange(0, buffer.pos, dtype=np.int64),
                ]
            )
        return np.arange(0, size, dtype=np.int64)

    keep = max(1, int(max_entries))
    if buffer.full:
        start = (buffer.pos - keep) % buffer.buffer_size
        if start < buffer.pos:
            return np.arange(start, buffer.pos, dtype=np.int64)
        return np.concatenate(
            [
                np.arange(start, buffer.buffer_size, dtype=np.int64),
                np.arange(0, buffer.pos, dtype=np.int64),
            ]
        )

    start = max(0, size - keep)
    return np.arange(start, size, dtype=np.int64)


def save_replay_buffer_snapshot(model, path, max_entries):
    assert model.replay_buffer is not None
    buffer = model.replay_buffer
    indices = _latest_replay_indices(buffer, max_entries)
    size = len(indices)

    snapshot = ReplayBuffer(
        buffer_size=max(size * buffer.n_envs, buffer.n_envs),
        observation_space=buffer.observation_space,
        action_space=buffer.action_space,
        device=buffer.device,
        n_envs=buffer.n_envs,
        optimize_memory_usage=False,
        handle_timeout_termination=buffer.handle_timeout_termination,
    )
    if size > 0:
        snapshot.observations[:size] = buffer.observations[indices]
        if buffer.optimize_memory_usage:
            next_indices = (indices + 1) % buffer.buffer_size
            snapshot.next_observations[:size] = buffer.observations[next_indices]
        else:
            snapshot.next_observations[:size] = buffer.next_observations[indices]
        snapshot.actions[:size] = buffer.actions[indices]
        snapshot.rewards[:size] = buffer.rewards[indices]
        snapshot.dones[:size] = buffer.dones[indices]
        snapshot.timeouts[:size] = buffer.timeouts[indices]
    snapshot.pos = size % snapshot.buffer_size
    snapshot.full = size == snapshot.buffer_size
    snapshot.optimize_memory_usage = False
    save_to_pkl(path, snapshot, 0)


def main():
    parser = argparse.ArgumentParser()
    default_asset_root = str(
        REPO_ROOT
        / "assets" / "external" / "interbotix_ros_manipulators" / "interbotix_ros_xsarms"
        / "interbotix_xsarm_descriptions" / "meshes"
    )
    parser.add_argument("--asset-root", default=default_asset_root)
    parser.add_argument("--timesteps", type=int, default=1_500_000)
    parser.add_argument("--run-dir", default="runs/wx250_image_sac_pixel_encoder")
    parser.add_argument("--drive-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--n-envs", type=int, default=1)
    parser.add_argument("--obs-size", type=int, default=64)
    parser.add_argument("--obs-camera", choices=WX250PickPlaceImageEnv.OBS_CAMERAS, default="front")
    parser.add_argument("--n-stack", type=int, default=3)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--saved-replay-buffer-size", type=int, default=15_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--feature-dim", type=int, default=50)
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--easy-reset-prob", type=float, default=0.0)
    parser.add_argument("--easy-distance-scale", type=float, default=0.5)
    parser.add_argument("--wandb-dir", default=None)
    parser.add_argument("--checkpoint-freq", type=int, default=50_000)
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--no-replay-buffer", action="store_true")
    args = parser.parse_args()

    storage_root = Path(args.drive_dir) if args.drive_dir is not None else Path(".")
    storage_root.mkdir(parents=True, exist_ok=True)
    run_dir = storage_root / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir = Path(args.wandb_dir) if args.wandb_dir else Path("/tmp/wandb")
    wandb_dir.mkdir(parents=True, exist_ok=True)

    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError("Set the WANDB_API_KEY environment variable before running.")
    os.environ["WANDB_DIR"] = str(wandb_dir.resolve())

    checkpoint_path = None
    if args.resume:
        ckpts = glob.glob(str(run_dir / "checkpoints" / "wx250_image_sac_pixel_encoder_*_steps.zip"))

        def _ckpt_step(path):
            match = re.search(r"_(\d+)_steps", Path(path).stem)
            return int(match.group(1)) if match else -1

        ckpts.sort(key=_ckpt_step)
        if ckpts:
            checkpoint_path = ckpts[-1]
        elif (run_dir / "final_model.zip").exists():
            checkpoint_path = str(run_dir / "final_model.zip")
        if checkpoint_path:
            print(f"Resuming from {checkpoint_path}")

    domain_randomize = not args.no_randomize
    n_envs = max(1, args.n_envs)
    vec_cls = DummyVecEnv if n_envs == 1 else SubprocVecEnv
    env = None
    eval_env = None

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

        if vec_cls is SubprocVecEnv:
            env = SubprocVecEnv(env_fns, start_method="spawn")
        else:
            env = DummyVecEnv(env_fns)
        eval_env = DummyVecEnv(eval_env_fns)

        env = VecFrameStack(env, n_stack=args.n_stack)
        eval_env = VecFrameStack(eval_env, n_stack=args.n_stack)
        gradient_steps = -1 if n_envs > 1 else 1

        wandb.init(
            project="wx250-pick-place",
            config={
                "algorithm": "SAC",
                "policy": "CnnPolicy",
                "obs_mode": "image",
                "encoder": "custom_pixel_encoder",
                "obs_size": args.obs_size,
                "obs_camera": args.obs_camera,
                "n_stack": args.n_stack,
                "n_envs": n_envs,
                "domain_randomize": domain_randomize,
                "easy_reset_prob": args.easy_reset_prob,
                "easy_distance_scale": args.easy_distance_scale,
                "feature_dim": args.feature_dim,
                "hidden_dim": args.hidden_dim,
                "learning_rate": args.learning_rate,
                "buffer_size": args.buffer_size,
                "saved_replay_buffer_size": args.saved_replay_buffer_size,
                "learning_starts": args.learning_starts,
                "batch_size": args.batch_size,
                "tau": 0.005,
                "gamma": 0.99,
                "gradient_steps": gradient_steps,
                "checkpoint_freq": args.checkpoint_freq,
                "total_timesteps": args.timesteps,
            },
            sync_tensorboard=False,
            dir=str(wandb_dir.resolve()),
            settings=wandb.Settings(init_timeout=300),
        )

        callbacks = [
            WandbEvalCallback(
                eval_env,
                best_model_save_path=str(run_dir / "best_model"),
                log_path=str(run_dir / "eval_logs"),
                eval_freq=max(10_000 // n_envs, 2_500),
                deterministic=True,
                render=False,
                n_eval_episodes=10,
            ),
            WandbMetricsCallback(log_freq=max(1000, n_envs * 500)),
        ]
        if not args.no_checkpoints:
            callbacks.insert(
                0,
                PruningCheckpointCallback(
                    save_freq=max(args.checkpoint_freq // n_envs, 1),
                    save_path=str(run_dir / "checkpoints"),
                    name_prefix="wx250_image_sac_pixel_encoder",
                    saved_replay_buffer_size=None if args.no_replay_buffer else args.saved_replay_buffer_size,
                ),
            )

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
                policy_kwargs=dict(
                    features_extractor_class=PixelEncoder,
                    features_extractor_kwargs=dict(features_dim=args.feature_dim),
                    net_arch=[args.hidden_dim, args.hidden_dim],
                    normalize_images=True,
                ),
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
            save_replay_buffer_snapshot(model, run_dir / "final_replay_buffer.pkl", args.saved_replay_buffer_size)
    finally:
        if eval_env is not None:
            eval_env.close()
        if env is not None:
            env.close()
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
