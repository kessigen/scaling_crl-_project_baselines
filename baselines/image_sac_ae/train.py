
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

from baselines.image_sac_ae import sac_ae_utils as utils
from baselines.image_sac_ae.chw_wrapper import ImageToCHW
from baselines.image_sac_ae.sac_ae_agent import SACAEAgent, UpdateMetrics
from baselines.image_sac_ae.sac_ae_replay_buffer import ReplayBuffer
from envs.wx250_pick_env_image import WX250PickPlaceImageEnv


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
    easy_reset_prob,
    easy_distance_scale,
    render_mode = None,
):
    env = WX250PickPlaceImageEnv(
        render_mode=render_mode,
        obs_mode="image",
        obs_height=obs_size,
        obs_width=obs_size,
        domain_randomize=domain_randomize,
        obs_camera=obs_camera,
        seed=seed,
        easy_reset_prob=easy_reset_prob,
        easy_distance_scale=easy_distance_scale,
    )
    return ImageToCHW(env)


def evaluate(agent, env, episodes):
    success = 0
    returns = []
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
        success += int(bool(last_info.get("is_success", False)))
        returns.append(total_reward)
    return float(np.mean(returns)), success / episodes


def save_checkpoint(
    path,
    agent,
    replay_buffer,
    step,
    episode,
    config,
    saved_replay_buffer_size = None,
    include_replay_buffer = True,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "agent": agent.state_dict(),
        "step": step,
        "episode": episode,
        "config": config,
    }
    torch.save(payload, path)
    if include_replay_buffer:
        replay_path = replay_buffer_path_for_checkpoint(path)
        replay_buffer.save(replay_path, max_entries=saved_replay_buffer_size)
        if path.parent.name == "checkpoints":
            prune_replay_snapshots(replay_path)


def load_checkpoint(path, agent, replay_buffer):
    payload = torch.load(path, map_location=agent.device, weights_only=False)
    agent.load_state_dict(payload["agent"])
    replay_path = replay_buffer_path_for_checkpoint(path)
    if replay_path.exists():
        replay_buffer.load(replay_path)
    elif "replay_buffer" in payload:
        # Backward compatibility with older checkpoints that embedded the replay state.
        replay_buffer.load_state_dict(payload["replay_buffer"])
    else:
        print(f"Warning: no replay buffer found in {path}; resuming weights only.")
    return int(payload["step"]), int(payload["episode"]), dict(payload.get("config", {}))


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


def log_to_wandb(
    step,
    train_start_time,
    last_log_step,
    last_log_time,
    recent_returns,
    recent_lengths,
    recent_successes,
    recent_durations,
    latest_metrics,
):
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
        log["rollout/ep_seconds_mean"] = float(np.mean(recent_durations))
        recent_returns.clear()
        recent_lengths.clear()
        recent_successes.clear()
        recent_durations.clear()

    if latest_metrics is not None:
        log["train/batch_reward"] = latest_metrics.batch_reward
        log["train/critic_loss"] = latest_metrics.critic_loss
        log["train/recon_loss"] = latest_metrics.recon_loss
        log["train/latent_loss"] = latest_metrics.latent_loss
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
    parser.add_argument("--run-dir", default="runs/wx250_image_sac_ae")
    parser.add_argument("--obs-size", type=int, default=64)
    parser.add_argument("--obs-camera", choices=WX250PickPlaceImageEnv.OBS_CAMERAS, default="front")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--saved-replay-buffer-size", type=int, default=20_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--feature-dim", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--decoder-lr", type=float, default=None)
    parser.add_argument("--discount", type=float, default=0.99)
    parser.add_argument("--critic-tau", type=float, default=0.01)
    parser.add_argument("--init-temperature", type=float, default=0.1)
    parser.add_argument("--actor-update-frequency", type=int, default=2)
    parser.add_argument("--critic-target-update-frequency", type=int, default=2)
    parser.add_argument("--decoder-latent-lambda", type=float, default=1e-6)
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
    parser.add_argument("--easy-reset-prob", type=float, default=0.0)
    parser.add_argument("--easy-distance-scale", type=float, default=0.5)
    parser.add_argument("--wandb-dir", default=None)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--resume", type=str, default=None, help="Path to a saved .pt checkpoint")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    checkpoints_dir = run_dir / "checkpoints"
    checkpoint_freq = args.checkpoint_freq if args.checkpoint_freq is not None else args.save_freq
    run_dir.mkdir(parents=True, exist_ok=True)
    utils.set_seed(args.seed)

    config = {
        "algorithm": "SAC+AE",
        "obs_mode": "image",
        "obs_size": args.obs_size,
        "obs_camera": args.obs_camera,
        "n_stack": 1,
        "device": args.device,
        "seed": args.seed,
        "buffer_size": args.buffer_size,
        "saved_replay_buffer_size": args.saved_replay_buffer_size,
        "batch_size": args.batch_size,
        "feature_dim": args.feature_dim,
        "hidden_dim": args.hidden_dim,
        "hidden_depth": args.hidden_depth,
        "lr": args.lr,
        "decoder_lr": args.decoder_lr if args.decoder_lr is not None else args.lr,
        "discount": args.discount,
        "critic_tau": args.critic_tau,
        "init_temperature": args.init_temperature,
        "actor_update_frequency": args.actor_update_frequency,
        "critic_target_update_frequency": args.critic_target_update_frequency,
        "decoder_latent_lambda": args.decoder_latent_lambda,
        "seed_steps": args.seed_steps,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "checkpoint_freq": checkpoint_freq,
        "timesteps": args.timesteps,
        "randomize_train": args.randomize_train,
        "randomize_eval": args.randomize_eval,
        "easy_reset_prob": args.easy_reset_prob,
        "easy_distance_scale": args.easy_distance_scale,
        "no_replay_buffer": args.no_replay_buffer,
    }

    use_wandb = not args.no_wandb
    wandb_id_file = run_dir / "wandb_run_id.txt"
    wandb_dir = Path(args.wandb_dir) if args.wandb_dir else Path("/tmp/wandb")
    if use_wandb:
        global wandb
        import wandb as wandb_module

        wandb = wandb_module
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("Set the WANDB_API_KEY environment variable before running, or pass --no-wandb.")
        wandb_dir.mkdir(parents=True, exist_ok=True)
        os.environ["WANDB_DIR"] = str(wandb_dir.resolve())
        wandb_id = wandb_id_file.read_text(encoding="utf-8").strip() if args.resume and wandb_id_file.exists() else None
        run = wandb.init(
            project="wx250-pick-place",
            config=config,
            sync_tensorboard=False,
            dir=str(wandb_dir.resolve()),
            settings=wandb.Settings(init_timeout=300),
            id=wandb_id,
            resume="allow" if wandb_id is not None else None,
        )
        wandb_id_file.write_text(run.id, encoding="utf-8")

    train_env = make_env(
        seed=args.seed,
        obs_size=args.obs_size,
        domain_randomize=args.randomize_train,
        obs_camera=args.obs_camera,
        easy_reset_prob=args.easy_reset_prob,
        easy_distance_scale=args.easy_distance_scale,
    )
    eval_env = make_env(
        seed=args.seed + 100,
        obs_size=args.obs_size,
        domain_randomize=args.randomize_eval,
        obs_camera=args.obs_camera,
        easy_reset_prob=0.0,
        easy_distance_scale=args.easy_distance_scale,
    )

    obs_shape = train_env.observation_space.shape
    action_shape = train_env.action_space.shape
    agent = SACAEAgent(
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
        decoder_lr=args.decoder_lr,
        actor_update_frequency=args.actor_update_frequency,
        critic_tau=args.critic_tau,
        critic_target_update_frequency=args.critic_target_update_frequency,
        decoder_latent_lambda=args.decoder_latent_lambda,
    )
    replay_buffer = ReplayBuffer(obs_shape, action_shape, args.buffer_size, args.device)

    start_step = 0
    episode = 0
    if args.resume:
        start_step, episode, saved_config = load_checkpoint(Path(args.resume), agent, replay_buffer)
        print(f"Resumed from {args.resume} at step={start_step} episode={episode}")
        if saved_config:
            print(f"Loaded saved config summary: feature_dim={saved_config.get('feature_dim')} hidden_dim={saved_config.get('hidden_dim')}")

    obs, _ = train_env.reset(seed=args.seed)
    episode_return = 0.0
    episode_step = 0
    episode_start_time = time.time()
    train_start_time = time.time()
    best_eval_reward = -float("inf")
    recent_returns = []
    recent_lengths = []
    recent_successes = []
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
                eval_reward, eval_success = evaluate(agent, eval_env, args.eval_episodes)
                print(f"[eval] step={step:7d} reward={eval_reward:8.3f} success={eval_success:6.1%}")
                append_csv(
                    run_dir / "eval_metrics.csv",
                    {
                        "step": step,
                        "eval_reward": eval_reward,
                        "eval_success": eval_success,
                    },
                )
                if use_wandb:
                    wandb.log(
                        {
                            "eval/mean_reward": eval_reward,
                            "eval/success_rate": eval_success,
                        },
                        step=step,
                    )
                if eval_reward > best_eval_reward:
                    best_eval_reward = eval_reward
                    best_checkpoint = run_dir / "best_model.pt"
                    save_checkpoint(
                        best_checkpoint,
                        agent,
                        replay_buffer,
                        step,
                        episode,
                        config,
                        saved_replay_buffer_size=args.saved_replay_buffer_size,
                        include_replay_buffer=not args.no_replay_buffer,
                    )
                    copy_checkpoint(best_checkpoint, run_dir / "best_model" / "best_model.pt")

            if not args.no_checkpoints and step % checkpoint_freq == 0:
                save_checkpoint(
                    checkpoints_dir / f"image_sac_ae_step_{step}.pt",
                    agent,
                    replay_buffer,
                    step,
                    episode,
                    config,
                    saved_replay_buffer_size=args.saved_replay_buffer_size,
                    include_replay_buffer=not args.no_replay_buffer,
                )

            if use_wandb and step % args.log_freq == 0:
                last_log_step, last_log_time = log_to_wandb(
                    step=step,
                    train_start_time=train_start_time,
                    last_log_step=last_log_step,
                    last_log_time=last_log_time,
                    recent_returns=recent_returns,
                    recent_lengths=recent_lengths,
                    recent_successes=recent_successes,
                    recent_durations=recent_durations,
                    latest_metrics=latest_metrics,
                )

            if done:
                duration = time.time() - episode_start_time
                success = int(bool(info.get("is_success", False)))
                recent_returns.append(episode_return)
                recent_lengths.append(episode_step)
                recent_successes.append(float(success))
                recent_durations.append(duration)

                log_row = {
                    "episode": episode,
                    "step": step,
                    "episode_return": episode_return,
                    "episode_length": episode_step,
                    "success": success,
                    "seconds": duration,
                }
                if metrics is not None:
                    log_row.update(
                        {
                            "batch_reward": metrics.batch_reward,
                            "critic_loss": metrics.critic_loss,
                            "actor_loss": metrics.actor_loss if metrics.actor_loss is not None else float("nan"),
                            "alpha_loss": metrics.alpha_loss if metrics.alpha_loss is not None else float("nan"),
                            "alpha_value": metrics.alpha_value,
                            "entropy": metrics.entropy if metrics.entropy is not None else float("nan"),
                            "recon_loss": metrics.recon_loss,
                            "latent_loss": metrics.latent_loss,
                        }
                    )
                append_csv(run_dir / "train_metrics.csv", log_row)
                print(
                    f"[train] ep={episode:4d} step={step:7d} return={episode_return:8.3f} "
                    f"len={episode_step:3d} success={bool(info.get('is_success', False))} "
                    f"alpha={metrics.alpha_value if metrics is not None else float('nan'):.4f} "
                    f"recon={metrics.recon_loss if metrics is not None else float('nan'):.5f}"
                )

                episode += 1
                obs, _ = train_env.reset()
                episode_return = 0.0
                episode_step = 0
                episode_start_time = time.time()

        save_checkpoint(
            run_dir / "final_model.pt",
            agent,
            replay_buffer,
            args.timesteps,
            episode,
            config,
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
