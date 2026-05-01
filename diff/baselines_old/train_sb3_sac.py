from __future__ import annotations

import argparse
import glob
from pathlib import Path
import re

import numpy as np
import wandb

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from wx250_pick_env import WX250PickPlaceEnv


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
                log["rollout/ep_reward_mean"] = np.mean(self.episode_rewards)
                log["rollout/ep_reward_min"] = np.min(self.episode_rewards)
                log["rollout/ep_reward_max"] = np.max(self.episode_rewards)
                log["rollout/ep_len_mean"] = np.mean(self.episode_lengths)
                self.episode_rewards.clear()
                self.episode_lengths.clear()

            if self.episode_successes:
                log["rollout/success_rate"] = np.mean(self.episode_successes)
                log["rollout/total_successes"] = self.total_successes
                self.episode_successes.clear()

            if self.episode_final_reach:
                log["rollout/final_reach_dist"] = np.mean(self.episode_final_reach)
                self.episode_final_reach.clear()

            if self.episode_final_place:
                log["rollout/final_place_dist"] = np.mean(self.episode_final_place)
                self.episode_final_place.clear()

            for key, vals in self.reward_components.items():
                if vals:
                    log[f"reward/{key}"] = np.mean(vals)
                    vals.clear()

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


def make_env(asset_root: str, render_mode=None):
    def _thunk():
        return Monitor(WX250PickPlaceEnv(render_mode=render_mode, asset_root=asset_root))
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
    default_asset_root = str(Path(__file__).resolve().parent / "external" / "interbotix_ros_manipulators" / "interbotix_ros_xsarms" / "interbotix_xsarm_descriptions" / "meshes")
    parser.add_argument("--asset-root", default=default_asset_root)
    parser.add_argument("--timesteps", type=int, default=500_000)
    parser.add_argument("--run-dir", default="runs/wx250_sac_mesh")
    parser.add_argument(
        "--drive-dir",
        default=None,
        help="Optional Google Drive root for persistent Colab outputs, e.g. /content/drive/MyDrive/wx250_runs",
    )
    parser.add_argument("--resume", action="store_true", help="Resume from latest checkpoint")
    args = parser.parse_args()

    storage_root = Path(args.drive_dir) if args.drive_dir is not None else Path(".")
    storage_root.mkdir(parents=True, exist_ok=True)
    run_dir = storage_root / args.run_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    wandb_dir = storage_root / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)

    import os
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
        ckpts = glob.glob(str(run_dir / "checkpoints" / "wx250_sac_*_steps.zip"))
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

    wandb.init(
        project="wx250-pick-place",
        config={
            "algorithm": "SAC",
            "net_arch": [256, 256, 256],
            "learning_rate": 1e-4,
            "buffer_size": 1_000_000,
            "batch_size": 256,
            "tau": 0.005,
            "gamma": 0.99,
            "total_timesteps": args.timesteps,
        },
        sync_tensorboard=False,
        dir=str(wandb_dir.resolve()),
    )
    wandb_id_file.write_text(wandb.run.id)

    env = DummyVecEnv([make_env(args.asset_root)])
    eval_env = DummyVecEnv([make_env(args.asset_root)])

    checkpoint_cb = CheckpointCallback(save_freq=135_000, save_path=str(run_dir / "checkpoints"), name_prefix="wx250_sac", save_replay_buffer=True)
    eval_cb = WandbEvalCallback(
        eval_env,
        best_model_save_path=str(run_dir / "best_model"),
        log_path=str(run_dir / "eval_logs"),
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=10,
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
            "MlpPolicy",
            env,
            policy_kwargs=dict(net_arch=[256, 256, 256]),
            learning_rate=1e-4,
            buffer_size=1_000_000,
            learning_starts=5_000,
            batch_size=256,
            tau=0.005,
            gamma=0.99,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto_0.01",
            verbose=1,
        )
    wandb_cb = WandbMetricsCallback(log_freq=1000)
    model.learn(total_timesteps=args.timesteps, callback=[checkpoint_cb, eval_cb, wandb_cb], progress_bar=True, reset_num_timesteps=checkpoint_path is None)
    model.save(str(run_dir / "final_model"))
    model.save_replay_buffer(str(run_dir / "final_replay_buffer"))
    wandb.finish()


if __name__ == "__main__":
    main()
