"""pose-SAC trainer for the in-zone-success wx250s env.

this is the "SAC (MLP)" baseline we report in the report (Table 1, pose
representation). plain SB3 SAC + MlpPolicy on pose observation from env
 since the env already gives us joint pos/vel + ee pos +
cube/goal vector + gripper.

two non-default changes from intial early DrQ runs:
  - init temperature 0.1 (ent_coef="auto_0.1"). SB3's default of 1.0 
  too noisy on pose obs, the policy seemed random for the
    first ~50k steps before getting any signal.
  - alpha floor at 0.05  without it SB3's auto-tune
    drives alpha to 0 around step 80k and becomes fully deterministic 
    same thing in most variant runs, same fix.

"""

import argparse
import glob
import math
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")


import numpy as np
import torch

from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.wx250_pick_env_in_zone import WX250PickPlaceImageInZoneEnv


wandb = None  # filled in by main() unless --no-wandb is passed


# adds a floor to alpha so it doesnt go to 0.
# without this i kept seeing the policy go fully deterministic around 80k
# steps and just stop exploring. floor of 0.05 works

class AlphaFloorCallback(BaseCallback):
    def __init__(self, min_alpha = 0.05):
        super().__init__()
        self.log_min = math.log(min_alpha)

    def _on_step(self):
        log_ent = getattr(self.model, "log_ent_coef", None)
        if log_ent is not None:
            with torch.no_grad():
                log_ent.data.clamp_(min=self.log_min)
        return True


class WandbMetricsCallback(BaseCallback):
    def __init__(self, log_freq = 1000):
        super().__init__()
        self.log_freq = log_freq
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_successes = []
        self.episode_in_zone_fractions = []
        self.total_successes = 0
        self.reward_components = {}

    def _on_step(self):
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])
        for i, info in enumerate(infos):
            bd = info.get("reward_breakdown", {})
            for key, val in bd.items():
                self.reward_components.setdefault(key, []).append(float(val))
            if dones[i]:
                # In-zone semantics: is_success is the per-step in-zone indicator,
                # so reading it on `done` gives in-zone-at-end.
                success = bool(info.get("is_success", False))
                self.episode_successes.append(float(success))
                if success:
                    self.total_successes += 1
                self.episode_in_zone_fractions.append(float(info.get("in_zone_fraction", 0.0)))
            if "episode" in info:
                self.episode_rewards.append(float(info["episode"]["r"]))
                self.episode_lengths.append(int(info["episode"]["l"]))

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
            if self.episode_in_zone_fractions:
                log["rollout/in_zone_fraction_mean"] = float(np.mean(self.episode_in_zone_fractions))
                self.episode_in_zone_fractions.clear()
            for key, vals in self.reward_components.items():
                if vals:
                    log[f"reward/{key}"] = float(np.mean(vals))
                    vals.clear()

            if self.model.logger is not None:
                lv = self.model.logger.name_to_value
                for key in (
                    "train/actor_loss",
                    "train/critic_loss",
                    "train/ent_coef",
                    "train/ent_coef_loss",
                    "train/learning_rate",
                ):
                    if key in lv:
                        log[key] = float(lv[key])

            if log and wandb is not None:
                wandb.log(log, step=self.num_timesteps)
        return True



class WandbEvalCallback(EvalCallback):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._eval_in_zone_fractions = []

    def _log_success_callback(self, locals_, globals_):
        super()._log_success_callback(locals_, globals_)
        if locals_.get("done"):
            info = locals_.get("info", {})
            if "in_zone_fraction" in info:
                self._eval_in_zone_fractions.append(float(info["in_zone_fraction"]))

    def _on_step(self):
        result = super()._on_step()
        if (
            wandb is not None
            and self.last_mean_reward is not None
            and self.num_timesteps % self.eval_freq == 0
        ):
            log = {
                "eval/mean_reward": float(self.last_mean_reward),
                "eval/best_mean_reward": float(self.best_mean_reward),
            }
            if self.evaluations_results:
                last_rewards = np.asarray(self.evaluations_results[-1], dtype=np.float64)
                log["eval/std_reward"] = float(np.std(last_rewards))
            if self.evaluations_length:
                last_lengths = np.asarray(self.evaluations_length[-1], dtype=np.float64)
                log["eval/mean_ep_length"] = float(np.mean(last_lengths))
            if self._is_success_buffer:
                log["eval/success_rate"] = float(np.mean(self._is_success_buffer))
            if self._eval_in_zone_fractions:
                log["eval/in_zone_fraction"] = float(np.mean(self._eval_in_zone_fractions))
                self._eval_in_zone_fractions.clear()
            wandb.log(log, step=self.num_timesteps)
        return result


def make_env(seed, env_kwargs, randomize):
    def _thunk():
        env = WX250PickPlaceImageInZoneEnv(
            obs_mode="pose",
            domain_randomize=randomize,
            seed=seed,
            **env_kwargs,
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
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--run-dir", default="runs/wx250s_in_zone_pose_sac")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--buffer-size", type=int, default=250_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--tau", type=float, default=0.005)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--init-temperature", type=float, default=0.1)
    parser.add_argument("--alpha-floor", type=float, default=0.05)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--randomize-train", action="store_true")
    parser.add_argument("--randomize-eval", action="store_true")
    # In-zone env flags (defaults match baselines/sac_drq).
    parser.add_argument("--red-zone-x", type=float, default=0.30)
    parser.add_argument("--red-zone-y", type=float, default=-0.12)
    parser.add_argument("--blue-zone-x", type=float, default=0.30)
    parser.add_argument("--blue-zone-y", type=float, default=0.12)
    parser.add_argument("--zone-half", type=float, default=0.06)
    parser.add_argument("--spawn-noise", type=float, default=0.02)
    parser.add_argument("--goal-noise", type=float, default=0.02)
    parser.add_argument("--success-threshold", type=float, default=0.04)
    parser.add_argument("--park-bonus", type=float, default=2.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-dir", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = run_dir / "checkpoints"

    use_wandb = not args.no_wandb
    if use_wandb:
        global wandb
        import wandb as wandb_module

        wandb = wandb_module
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("Set WANDB_API_KEY before running, or pass --no-wandb.")
        wandb_dir = Path(args.wandb_dir) if args.wandb_dir else Path("/tmp/wandb")
        wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(wandb_dir.resolve())

        wandb_id_file = run_dir / "wandb_run_id.txt"
        wandb_id = None
        if args.resume and wandb_id_file.exists():
            wandb_id = wandb_id_file.read_text(encoding="utf-8").strip()

        run = wandb.init(
            project="wx250s-in-zone-pose-sac",
            config={
                "algorithm": "SAC",
                "obs_mode": "pose",
                "net_arch": [256, 256],
                "learning_rate": args.lr,
                "buffer_size": args.buffer_size,
                "batch_size": args.batch_size,
                "tau": args.tau,
                "gamma": args.gamma,
                "learning_starts": args.learning_starts,
                "init_temperature": args.init_temperature,
                "alpha_floor": args.alpha_floor,
                "target_entropy": "auto (-action_dim, stiff)",
                "timesteps": args.timesteps,
                "max_steps": args.max_steps,
                "red_zone": (args.red_zone_x, args.red_zone_y),
                "blue_zone": (args.blue_zone_x, args.blue_zone_y),
                "zone_half": args.zone_half,
                "spawn_noise": args.spawn_noise,
                "goal_noise": args.goal_noise,
                "success_threshold": args.success_threshold,
                "park_bonus": args.park_bonus,
                "randomize_train": args.randomize_train,
                "randomize_eval": args.randomize_eval,
                "seed": args.seed,
                "device": args.device,
            },
            sync_tensorboard=False,
            dir=str(wandb_dir.resolve()),
            id=wandb_id,
            resume="allow" if wandb_id is not None else None,
        )
        wandb_id_file.write_text(run.id, encoding="utf-8")

    env_kwargs = dict(
        red_zone_center_xy=(args.red_zone_x, args.red_zone_y),
        blue_zone_center_xy=(args.blue_zone_x, args.blue_zone_y),
        zone_half_extent_xy=args.zone_half,
        spawn_noise_xy=args.spawn_noise,
        goal_noise_xy=args.goal_noise,
        success_threshold_xy=args.success_threshold,
        park_bonus=args.park_bonus,
        max_steps=args.max_steps,
    )
    train_env = DummyVecEnv([make_env(args.seed, env_kwargs, args.randomize_train)])
    eval_env = DummyVecEnv([make_env(args.seed + 100, env_kwargs, args.randomize_eval)])

    action_dim = int(train_env.action_space.shape[0])

    checkpoint_path = None
    if args.resume:
        ckpts = glob.glob(str(checkpoints_dir / "wx250_pose_sac_*_steps.zip"))

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

    if checkpoint_path:
        model = SAC.load(checkpoint_path, env=train_env, device=args.device)
        replay_buffer_path = replay_buffer_path_from_checkpoint(checkpoint_path)
        if replay_buffer_path is not None and replay_buffer_path.exists():
            model.load_replay_buffer(str(replay_buffer_path))
        else:
            print(
                f"Warning: no replay buffer alongside {checkpoint_path}; "
                "resuming weights only."
            )
    else:
        model = SAC(
            "MlpPolicy",
            train_env,
            policy_kwargs=dict(net_arch=[256, 256]),
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            learning_starts=args.learning_starts,
            batch_size=args.batch_size,
            tau=args.tau,
            gamma=args.gamma,
            train_freq=1,
            gradient_steps=1,
            ent_coef=f"auto_{args.init_temperature}",
            target_entropy=float(-action_dim),
            device=args.device,
            verbose=1,
        )

    callbacks = [AlphaFloorCallback(min_alpha=args.alpha_floor)]
    callbacks.append(
        CheckpointCallback(
            save_freq=args.save_freq,
            save_path=str(checkpoints_dir),
            name_prefix="wx250_pose_sac",
            save_replay_buffer=True,
        )
    )
    eval_cb_cls = WandbEvalCallback if use_wandb else EvalCallback
    callbacks.append(
        eval_cb_cls(
            eval_env,
            best_model_save_path=str(run_dir / "best_model"),
            log_path=str(run_dir / "eval_logs"),
            eval_freq=args.eval_freq,
            deterministic=True,
            render=False,
            n_eval_episodes=args.eval_episodes,
        )
    )
    if use_wandb:
        callbacks.append(WandbMetricsCallback(log_freq=args.log_freq))

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=checkpoint_path is None,
        )
        model.save(str(run_dir / "final_model"))
        model.save_replay_buffer(str(run_dir / "final_replay_buffer"))
    finally:
        train_env.close()
        eval_env.close()
        if use_wandb and wandb is not None and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
