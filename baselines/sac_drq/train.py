"""DrQ (image SAC trainer) for the in-zone-success wx250s env.

this is the "SAC + DRQ" baseline in the report (Table 1, image rep). same
training loop as baselines/image_drq/train.py - i didn't fork the agent,
just point this entry-point at the in-zone env so the zone / threshold /
park-bonus knobs are wired up. the actual DRQ agent + replay buffer +
frame-stack live in baselines/image_drq/ and are imported below.

Note:
  - replay buffer dump per checkpoint too big. (saw ~40GB per ckpt
    at buffer_size=500k-( use 15-20k. around 600MB in size)
  - switch to to iso camera
"""

import argparse
import csv
import os
import shutil
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")


import numpy as np
import torch

from baselines.image_drq import drq_utils as utils
from baselines.image_drq.drq_agent import DRQAgent, UpdateMetrics
from baselines.image_drq.drq_replay_buffer import ReplayBuffer
from baselines.image_drq.frame_stack import DrQFrameStack
from envs.wx250_pick_env_in_zone import WX250PickPlaceImageInZoneEnv


wandb = None


def replay_buffer_path_for_checkpoint(path):
    if path.stem == "final_model":
        return path.with_name("final_replay_buffer.pkl")
    return path.with_name(f"{path.stem}_replay_buffer.pkl")


def prune_replay_snapshots(current):
    for candidate in current.parent.glob("*_replay_buffer.pkl"):
        if candidate.resolve() != current.resolve():
            try:
                candidate.unlink()
            except OSError as exc:
                print(f"[replay-prune] failed to delete {candidate}: {exc}")


def make_env(
    seed,
    obs_size,
    domain_randomize,
    obs_camera,
    red_zone_xy,
    blue_zone_xy,
    zone_half,
    spawn_noise,
    goal_noise,
    success_threshold,
    park_bonus,
    max_steps,
    render_mode = None,
):
    env = WX250PickPlaceImageInZoneEnv(
        render_mode=render_mode,
        obs_mode="image",
        obs_height=obs_size,
        obs_width=obs_size,
        domain_randomize=domain_randomize,
        obs_camera=obs_camera,
        seed=seed,
        red_zone_center_xy=red_zone_xy,
        blue_zone_center_xy=blue_zone_xy,
        zone_half_extent_xy=zone_half,
        spawn_noise_xy=spawn_noise,
        goal_noise_xy=goal_noise,
        success_threshold_xy=success_threshold,
        park_bonus=park_bonus,
        max_steps=max_steps,
    )
    return DrQFrameStack(env, k=3)


def evaluate(agent, env, episodes):
    # rolls out `episodes` deterministic eps and returns
    # (mean_return, in_zone_at_end_rate, mean_in_zone_fraction).
    successes = 0
    returns = []
    in_zone_fractions = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        last_info = {}
        while not done:
            action = agent.act(obs, sample=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            done = bool(terminated or truncated)
            last_info = info
        successes += int(bool(last_info.get("is_success", False)))
        in_zone_fractions.append(float(last_info.get("in_zone_fraction", 0.0)))
        returns.append(total_reward)
    return float(np.mean(returns)), successes / episodes, float(np.mean(in_zone_fractions))


def save_checkpoint(
    path,
    agent,
    replay_buffer,
    step,
    episode,
    saved_replay_buffer_size = None,
    include_replay_buffer = True,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"agent": agent.state_dict(), "step": step, "episode": episode}
    torch.save(payload, path)
    if include_replay_buffer:
        replay_path = replay_buffer_path_for_checkpoint(path)
        replay_buffer.save(replay_path, max_entries=saved_replay_buffer_size)
        if path.parent.name == "checkpoints":
            prune_replay_snapshots(replay_path)


def load_checkpoint(path, agent, replay_buffer):
    payload = torch.load(path, map_location=agent.device)
    agent.load_state_dict(payload["agent"])
    replay_path = replay_buffer_path_for_checkpoint(path)
    if replay_path.exists():
        replay_buffer.load(replay_path)
    elif "replay_buffer" in payload:
        replay_buffer.load_state_dict(payload["replay_buffer"])
    else:
        print(f"Warning: no replay buffer found alongside {path}; resuming weights only.")
    return int(payload["step"]), int(payload["episode"])


def append_csv(path, row):
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if is_new:
            writer.writeheader()
        writer.writerow(row)


def copy_checkpoint(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    src_replay = replay_buffer_path_for_checkpoint(src)
    if src_replay.exists():
        shutil.copy2(src_replay, replay_buffer_path_for_checkpoint(dst))


def log_to_wandb(step, train_start_time, last_log_step, last_log_time,
                 recent_returns, recent_lengths, recent_successes,
                 recent_in_zone_fractions, recent_durations, latest_metrics):
    assert wandb is not None
    now = time.time()
    dstep = step - last_log_step
    dt = now - last_log_time

    log = {}
    if recent_returns:
        log["rollout/ep_reward_mean"] = float(np.mean(recent_returns))
        log["rollout/ep_reward_min"] = float(np.min(recent_returns))
        log["rollout/ep_reward_max"] = float(np.max(recent_returns))
        log["rollout/ep_len_mean"] = float(np.mean(recent_lengths))
        log["rollout/success_rate"] = float(np.mean(recent_successes))
        log["rollout/in_zone_fraction_mean"] = float(np.mean(recent_in_zone_fractions))
        log["rollout/ep_seconds_mean"] = float(np.mean(recent_durations))
        recent_returns.clear()
        recent_lengths.clear()
        recent_successes.clear()
        recent_in_zone_fractions.clear()
        recent_durations.clear()

    if latest_metrics is not None:
        log["train/batch_reward"] = latest_metrics.batch_reward
        log["train/critic_loss"] = latest_metrics.critic_loss
        log["train/alpha_value"] = latest_metrics.alpha_value
        if latest_metrics.actor_loss is not None:
            log["train/actor_loss"] = latest_metrics.actor_loss
        if latest_metrics.alpha_loss is not None:
            log["train/alpha_loss"] = latest_metrics.alpha_loss
        if latest_metrics.entropy is not None:
            log["train/entropy"] = latest_metrics.entropy

    if dstep > 0 and dt > 0:
        log["time/fps"] = dstep / dt
    log["time/total_seconds"] = now - train_start_time

    if log:
        wandb.log(log, step=step)
    return step, now


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--run-dir", default="runs/wx250s_in_zone_image_drq")
    parser.add_argument("--obs-size", type=int, default=64)
    parser.add_argument("--obs-camera", choices=WX250PickPlaceImageInZoneEnv.OBS_CAMERAS, default="front")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--buffer-size", type=int, default=500_000)
    parser.add_argument("--saved-replay-buffer-size", type=int, default=8_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--image-pad", type=int, default=4)
    parser.add_argument("--feature-dim", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--critic-tau", type=float, default=0.01)
    parser.add_argument("--init-temperature", type=float, default=0.1)
    parser.add_argument("--actor-update-frequency", type=int, default=2)
    parser.add_argument("--critic-target-update-frequency", type=int, default=2)
    parser.add_argument("--seed-steps", type=int, default=1000)
    parser.add_argument("--eval-freq", type=int, default=10_000)
    parser.add_argument("--eval-episodes", type=int, default=10)
    parser.add_argument("--save-freq", type=int, default=50_000)
    parser.add_argument("--log-freq", type=int, default=1000)
    parser.add_argument("--checkpoint-freq", type=int, default=None)
    parser.add_argument("--no-checkpoints", action="store_true")
    parser.add_argument("--no-replay-buffer", action="store_true")
    parser.add_argument("--randomize-train", action="store_true")
    parser.add_argument("--randomize-eval", action="store_true")
    # In-zone env knobs.
    parser.add_argument("--red-zone-x", type=float, default=0.30)
    parser.add_argument("--red-zone-y", type=float, default=-0.12)
    parser.add_argument("--blue-zone-x", type=float, default=0.30)
    parser.add_argument("--blue-zone-y", type=float, default=0.12)
    parser.add_argument("--zone-half", type=float, default=0.06,
                        help="Visible zone marker half-extent (m). Must be >= success-threshold.")
    parser.add_argument("--spawn-noise", type=float, default=0.02)
    parser.add_argument("--goal-noise", type=float, default=0.02)
    parser.add_argument("--success-threshold", type=float, default=0.04,
                        help="XY half-extent of the success region around the (jittered) goal (m).")
    parser.add_argument("--park-bonus", type=float, default=2.0,
                        help="Per-step reward while cube is inside the success region.")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--wandb-dir", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to a saved .pt checkpoint")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_freq = args.checkpoint_freq if args.checkpoint_freq is not None else args.save_freq
    run_dir.mkdir(parents=True, exist_ok=True)
    utils.set_seed(args.seed)

    use_wandb = not args.no_wandb
    wandb_id_file = run_dir / "wandb_run_id.txt"
    wandb_dir = Path(args.wandb_dir) if args.wandb_dir else Path("/tmp/wandb")
    if use_wandb:
        global wandb
        import wandb as wandb_module
        wandb = wandb_module
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("Set WANDB_API_KEY before running, or pass --no-wandb.")
        wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(wandb_dir.resolve())
        wandb_id = wandb_id_file.read_text(encoding="utf-8").strip() if args.resume and wandb_id_file.exists() else None
        run = wandb.init(
            project="wx250s-in-zone-drq",
            config={
                "algorithm": "DrQ",
                "obs_mode": "image",
                "obs_size": args.obs_size,
                "obs_camera": args.obs_camera,
                "n_stack": 3,
                "device": args.device,
                "seed": args.seed,
                "buffer_size": args.buffer_size,
                "saved_replay_buffer_size": args.saved_replay_buffer_size,
                "batch_size": args.batch_size,
                "image_pad": args.image_pad,
                "feature_dim": args.feature_dim,
                "hidden_dim": args.hidden_dim,
                "hidden_depth": args.hidden_depth,
                "lr": args.lr,
                "discount": args.discount,
                "critic_tau": args.critic_tau,
                "init_temperature": args.init_temperature,
                "actor_update_frequency": args.actor_update_frequency,
                "critic_target_update_frequency": args.critic_target_update_frequency,
                "seed_steps": args.seed_steps,
                "eval_freq": args.eval_freq,
                "eval_episodes": args.eval_episodes,
                "checkpoint_freq": checkpoint_freq,
                "timesteps": args.timesteps,
                "randomize_train": args.randomize_train,
                "randomize_eval": args.randomize_eval,
                "red_zone": (args.red_zone_x, args.red_zone_y),
                "blue_zone": (args.blue_zone_x, args.blue_zone_y),
                "zone_half": args.zone_half,
                "spawn_noise": args.spawn_noise,
                "goal_noise": args.goal_noise,
                "success_threshold": args.success_threshold,
                "park_bonus": args.park_bonus,
                "max_steps": args.max_steps,
                "no_replay_buffer": args.no_replay_buffer,
            },
            sync_tensorboard=False,
            dir=str(wandb_dir.resolve()),
            settings=wandb.Settings(init_timeout=300),
            id=wandb_id,
            resume="allow" if wandb_id is not None else None,
        )
        wandb_id_file.write_text(run.id, encoding="utf-8")

    env_kwargs = dict(
        obs_size=args.obs_size,
        obs_camera=args.obs_camera,
        red_zone_xy=(args.red_zone_x, args.red_zone_y),
        blue_zone_xy=(args.blue_zone_x, args.blue_zone_y),
        zone_half=args.zone_half,
        spawn_noise=args.spawn_noise,
        goal_noise=args.goal_noise,
        success_threshold=args.success_threshold,
        park_bonus=args.park_bonus,
        max_steps=args.max_steps,
    )
    train_env = make_env(seed=args.seed, domain_randomize=args.randomize_train, **env_kwargs)
    eval_env = make_env(seed=args.seed + 100, domain_randomize=args.randomize_eval, **env_kwargs)

    obs_shape = train_env.observation_space.shape
    action_shape = train_env.action_space.shape
    agent = DRQAgent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        action_range=(float(train_env.action_space.low.min()), float(train_env.action_space.high.max())),
        device=args.device,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
        discount=args.discount,
        init_temperature=args.init_temperature,
        lr=args.lr,
        actor_update_frequency=args.actor_update_frequency,
        critic_tau=args.critic_tau,
        critic_target_update_frequency=args.critic_target_update_frequency,
    )
    replay_buffer = ReplayBuffer(obs_shape, action_shape, args.buffer_size, args.image_pad, args.device)

    start_step = 0
    episode = 0
    if args.resume:
        start_step, episode = load_checkpoint(Path(args.resume), agent, replay_buffer)
        print(f"Resumed from {args.resume} at step={start_step} episode={episode}")

    obs, _ = train_env.reset(seed=args.seed)
    episode_return = 0.0
    episode_step = 0
    episode_start_time = time.time()
    train_start_time = time.time()
    best_eval_reward = -float("inf")
    recent_returns = []
    recent_lengths = []
    recent_successes = []
    recent_in_zone_fractions = []
    recent_durations = []
    last_log_step = start_step
    last_log_time = time.time()
    latest_metrics = None

    try:
        for step in range(start_step + 1, args.timesteps + 1):
            if step < args.seed_steps:
                action = train_env.action_space.sample()
            else:
                action = agent.act(obs, sample=True)

            next_obs, reward, terminated, truncated, info = train_env.step(action)
            done = bool(terminated or truncated)
            done_no_max = bool(terminated)
            replay_buffer.add(obs, action, float(reward), next_obs, done, done_no_max)

            obs = next_obs
            episode_return += float(reward)
            episode_step += 1

            metrics = None
            if step >= args.seed_steps and len(replay_buffer) >= args.batch_size:
                metrics = agent.update(replay_buffer, args.batch_size, step)
                latest_metrics = metrics

            if step % args.eval_freq == 0:
                eval_reward, eval_success, eval_in_zone = evaluate(agent, eval_env, args.eval_episodes)
                print(f"[eval] step={step:7d} reward={eval_reward:8.3f} "
                      f"success={eval_success:6.1%} in_zone_frac={eval_in_zone:5.2f}")
                append_csv(run_dir / "eval_metrics.csv", {
                    "step": step,
                    "eval_reward": eval_reward,
                    "eval_success": eval_success,
                    "eval_in_zone_fraction": eval_in_zone,
                })
                if use_wandb:
                    wandb.log({
                        "eval/mean_reward": eval_reward,
                        "eval/success_rate": eval_success,
                        "eval/in_zone_fraction": eval_in_zone,
                    }, step=step)
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    best_checkpoint = run_dir / "best_model.pt"
                    save_checkpoint(
                        best_checkpoint, agent, replay_buffer, step, episode,
                        saved_replay_buffer_size=args.saved_replay_buffer_size,
                        include_replay_buffer=not args.no_replay_buffer,
                    )
                    copy_checkpoint(best_checkpoint, run_dir / "best_model" / "best_model.pt")

            if not args.no_checkpoints and step % checkpoint_freq == 0:
                save_checkpoint(
                    checkpoints_dir / f"image_drq_step_{step}.pt",
                    agent, replay_buffer, step, episode,
                    saved_replay_buffer_size=args.saved_replay_buffer_size,
                    include_replay_buffer=not args.no_replay_buffer,
                )

            if use_wandb and step % args.log_freq == 0:
                last_log_step, last_log_time = log_to_wandb(
                    step=step, train_start_time=train_start_time,
                    last_log_step=last_log_step, last_log_time=last_log_time,
                    recent_returns=recent_returns, recent_lengths=recent_lengths,
                    recent_successes=recent_successes,
                    recent_in_zone_fractions=recent_in_zone_fractions,
                    recent_durations=recent_durations,
                    latest_metrics=latest_metrics,
                )

            if done:
                duration = time.time() - episode_start_time
                success = int(bool(info.get("is_success", False)))
                in_zone_frac = float(info.get("in_zone_fraction", 0.0))
                recent_returns.append(episode_return)
                recent_lengths.append(episode_step)
                recent_successes.append(float(success))
                recent_in_zone_fractions.append(in_zone_frac)
                recent_durations.append(duration)

                log_row = {
                    "episode": episode,
                    "step": step,
                    "episode_return": episode_return,
                    "episode_length": episode_step,
                    "success": success,
                    "in_zone_fraction": in_zone_frac,
                    "seconds": duration,
                }
                if metrics is not None:
                    log_row.update({
                        "batch_reward": metrics.batch_reward,
                        "critic_loss": metrics.critic_loss,
                        "actor_loss": metrics.actor_loss if metrics.actor_loss is not None else float("nan"),
                        "alpha_loss": metrics.alpha_loss if metrics.alpha_loss is not None else float("nan"),
                        "alpha_value": metrics.alpha_value,
                        "entropy": metrics.entropy if metrics.entropy is not None else float("nan"),
                    })
                append_csv(run_dir / "train_metrics.csv", log_row)
                print(f"[train] ep={episode:4d} step={step:7d} return={episode_return:8.3f} "
                      f"len={episode_step:3d} success={bool(success)} "
                      f"in_zone_frac={in_zone_frac:.2f} "
                      f"alpha={metrics.alpha_value if metrics is not None else float('nan'):.4f}")

                episode += 1
                obs, _ = train_env.reset()
                episode_return = 0.0
                episode_step = 0
                episode_start_time = time.time()

        save_checkpoint(
            run_dir / "final_model.pt",
            agent, replay_buffer, args.timesteps, episode,
            saved_replay_buffer_size=args.saved_replay_buffer_size,
            include_replay_buffer=not args.no_replay_buffer,
        )
    finally:
        train_env.close()
        eval_env.close()
        if use_wandb and wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
