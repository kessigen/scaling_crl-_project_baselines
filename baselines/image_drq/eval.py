
import argparse
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

if sys.platform.startswith("linux"):
    os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
from PIL import Image
import torch

from baselines.image_drq.drq_agent import DRQAgent
from baselines.image_drq.frame_stack import DrQFrameStack
from envs.wx250_pick_env_image import WX250PickPlaceImageEnv


def make_env(
    seed,
    obs_size,
    domain_randomize,
    obs_camera,
    goal_half,
    full_goal_success,
    legacy_reward,
    record,
):
    env = WX250PickPlaceImageEnv(
        render_mode="rgb_array" if record else None,
        obs_mode="image",
        obs_height=obs_size,
        obs_width=obs_size,
        domain_randomize=domain_randomize,
        obs_camera=obs_camera,
        goal_half=goal_half,
        full_goal_success=full_goal_success,
        legacy_reward=legacy_reward,
        seed=seed,
    )
    return DrQFrameStack(env, k=3)


def save_gif(frames, out, duration_ms = 50):
    pil_frames = [Image.fromarray(frame) for frame in frames]
    palette_frame = pil_frames[0].convert("P", palette=Image.ADAPTIVE, colors=255, dither=Image.NONE)
    quantized_frames = [palette_frame]
    quantized_frames.extend(frame.quantize(palette=palette_frame, dither=Image.NONE) for frame in pil_frames[1:])
    quantized_frames[0].save(out, save_all=True, append_images=quantized_frames[1:], duration=duration_ms, loop=0)


def select_episode_for_gif(episodes, mode):
    successful_only = mode.endswith("-success")
    if successful_only:
        episodes = [episode for episode in episodes if bool(episode["success"])]
        if not episodes:
            raise ValueError("No successful evaluated episodes were available for GIF recording.")
        mode = mode[: -len("-success")]

    if mode == "best":
        return max(episodes, key=lambda episode: float(episode["return"]))
    if mode == "random":
        return episodes[int(np.random.randint(len(episodes)))]

    median_return = float(np.median([float(episode["return"]) for episode in episodes]))
    return min(
        episodes,
        key=lambda episode: (abs(float(episode["return"]) - median_return), -float(episode["return"])),
    )


def write_episode_gif(out_path, selected_episode, label):
    print(
        f"\nSaving {label} "
        f"(ep {selected_episode['episode']}, return={selected_episode['return']:.2f}, "
        f"steps={selected_episode['steps']}, success={selected_episode['success']}) "
        f"to {out_path}..."
    )
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    frames = selected_episode["frames"]
    assert isinstance(frames, list)
    save_gif(frames, out)
    print(f"saved {len(frames)} frames to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--obs-size", type=int, default=64)
    parser.add_argument("--obs-camera", choices=WX250PickPlaceImageEnv.OBS_CAMERAS, default="front")
    parser.add_argument("--feature-dim", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--randomize-eval", action="store_true")
    parser.add_argument("--goal-half", type=float, default=0.07, help="Half-size of the visible green goal square in meters.")
    parser.add_argument("--full-goal-success", action="store_true", help="Require the full cube footprint to fit inside the green goal square.")
    parser.add_argument("--legacy-reward", action="store_true", help="Use the pre-rebalance staged reward. Must match the checkpoint's training reward.")
    parser.add_argument("--record-gif", type=str, default=None)
    parser.add_argument(
        "--record-gif-mode",
        choices=("random", "best", "median", "random-success", "best-success", "median-success"),
        default="random",
        help="Which evaluated episode to save when --record-gif is set. Modes ending in -success only consider successful episodes.",
    )
    parser.add_argument("--record-best-gif", type=str, default=None, help="If set, save the best-return evaluated episode to this .gif path.")
    parser.add_argument(
        "--record-success-gif",
        type=str,
        default=None,
        help="If set, save a successful evaluated episode to this .gif path.",
    )
    parser.add_argument(
        "--record-success-gif-mode",
        choices=("random-success", "best-success", "median-success"),
        default="random-success",
        help="Which successful evaluated episode to save when --record-success-gif is set.",
    )
    args = parser.parse_args()
    should_record_episodes = any((args.record_gif, args.record_best_gif, args.record_success_gif))

    env = make_env(
        seed=123,
        obs_size=args.obs_size,
        domain_randomize=args.randomize_eval,
        obs_camera=args.obs_camera,
        goal_half=args.goal_half,
        full_goal_success=args.full_goal_success,
        legacy_reward=args.legacy_reward,
        record=should_record_episodes,
    )
    agent = DRQAgent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        action_range=(float(env.action_space.low.min()), float(env.action_space.high.max())),
        device=args.device,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        hidden_depth=args.hidden_depth,
    )
    checkpoint = torch.load(args.model, map_location=agent.device)
    agent.load_state_dict(checkpoint["agent"])
    agent.train(False)

    successes = 0
    returns = []
    lengths = []
    recorded_episodes = []
    for ep in range(args.n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        last_info = {}
        frames = []
        if should_record_episodes:
            frame = env.unwrapped.render()
            if frame is not None:
                frames.append(frame.copy())
        while not done:
            action = agent.act(obs, sample=False)
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            steps += 1
            done = bool(terminated or truncated)
            last_info = info
            if should_record_episodes:
                frame = env.unwrapped.render()
                if frame is not None:
                    frames.append(frame.copy())
        success = bool(last_info.get("is_success", False))
        successes += int(success)
        returns.append(total_reward)
        lengths.append(steps)
        if should_record_episodes:
            recorded_episodes.append(
                {
                    "episode": ep + 1,
                    "return": total_reward,
                    "steps": steps,
                    "success": success,
                    "frames": frames,
                }
            )
        print(f"ep {ep + 1:3d}: return={total_reward:7.2f} steps={steps:3d} success={success}")

    print()
    print(f"success rate:   {successes}/{args.n_episodes} = {successes / args.n_episodes:.0%}")
    print(f"return mean+-std: {np.mean(returns):.2f} +- {np.std(returns):.2f}")
    print(f"length mean+-std: {np.mean(lengths):.1f} +- {np.std(lengths):.1f}")

    if recorded_episodes:
        best_episode = max(recorded_episodes, key=lambda episode: float(episode["return"]))
        print(
            f"best return:    ep {best_episode['episode']}  return={best_episode['return']:.2f}  "
            f"steps={best_episode['steps']}  success={best_episode['success']}"
        )
        successful_episodes = [episode for episode in recorded_episodes if bool(episode["success"])]
        if successful_episodes:
            best_success = max(successful_episodes, key=lambda episode: float(episode["return"]))
            print(
                f"best success:   ep {best_success['episode']}  return={best_success['return']:.2f}  "
                f"steps={best_success['steps']}"
            )

    if args.record_gif:
        selected_episode = select_episode_for_gif(recorded_episodes, args.record_gif_mode)
        write_episode_gif(args.record_gif, selected_episode, f"{args.record_gif_mode} evaluated episode")
    if args.record_best_gif:
        best_episode = select_episode_for_gif(recorded_episodes, "best")
        write_episode_gif(args.record_best_gif, best_episode, "best-return evaluated episode")
    if args.record_success_gif:
        success_episode = select_episode_for_gif(recorded_episodes, args.record_success_gif_mode)
        write_episode_gif(args.record_success_gif, success_episode, f"{args.record_success_gif_mode} evaluated episode")

    env.close()


if __name__ == "__main__":
    main()
