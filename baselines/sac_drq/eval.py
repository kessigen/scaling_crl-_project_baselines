"""eval + GIF recorder for the SAC+DRQ in-zone-success checkpoints.

loads a .pt saved by sac_drq/train.py (state_dict format), rolls it out
deterministically and prints the usual return / SR / in-zone-fraction
numbers. 

Note: USE same --obs-size / --obs-camera / --feature-dim /
--hidden-dim / --hidden-depthas in train, 

GIF recording uses env.unwrapped.render() (not the wrapped frame-stack) not the 64x64 obs.
"""

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
from envs.wx250_pick_env_in_zone import WX250PickPlaceImageInZoneEnv


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
    record,
):
    env = WX250PickPlaceImageInZoneEnv(
        render_mode="rgb_array" if record else None,
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


def save_gif(frames, out, duration_ms = 50):
    pil_frames = [Image.fromarray(frame) for frame in frames]
    palette_frame = pil_frames[0].convert("P", palette=Image.ADAPTIVE, colors=255, dither=Image.NONE)
    quantized = [palette_frame]
    quantized.extend(f.quantize(palette=palette_frame, dither=Image.NONE) for f in pil_frames[1:])
    quantized[0].save(out, save_all=True, append_images=quantized[1:], duration=duration_ms, loop=0)


def select_episode_for_gif(episodes, mode):
    successful_only = mode.endswith("-success")
    if successful_only:
        episodes = [ep for ep in episodes if bool(ep["success"])]
        if not episodes:
            raise ValueError("No successful evaluated episodes available for GIF recording.")
        mode = mode[: -len("-success")]
    if mode == "best":
        return max(episodes, key=lambda ep: float(ep["return"]))
    if mode == "random":
        return episodes[int(np.random.randint(len(episodes)))]
    median_ret = float(np.median([float(ep["return"]) for ep in episodes]))
    return min(episodes, key=lambda ep: (abs(float(ep["return"]) - median_ret), -float(ep["return"])))


def write_episode_gif(out_path, ep, label):
    print(f"\nSaving {label} (ep {ep['episode']}, return={ep['return']:.2f}, "
          f"steps={ep['steps']}, success={ep['success']}, "
          f"in_zone_frac={ep['in_zone_fraction']:.2f}) to {out_path}...")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_gif(ep["frames"], out)
    print(f"saved {len(ep['frames'])} frames to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .pt checkpoint")
    parser.add_argument("--n-episodes", type=int, default=20)
    parser.add_argument("--obs-size", type=int, default=64)
    parser.add_argument("--obs-camera", choices=WX250PickPlaceImageInZoneEnv.OBS_CAMERAS, default="front")
    parser.add_argument("--feature-dim", type=int, default=50)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--randomize-eval", action="store_true")
    # In-zone env knobs (must match training to be apples-to-apples).
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
    parser.add_argument("--record-gif", type=str, default=None)
    parser.add_argument("--record-gif-mode",
                        choices=("random", "best", "median", "random-success", "best-success", "median-success"),
                        default="random")
    parser.add_argument("--record-best-gif", type=str, default=None)
    parser.add_argument("--record-success-gif", type=str, default=None)
    parser.add_argument("--record-success-gif-mode",
                        choices=("random-success", "best-success", "median-success"),
                        default="random-success")
    args = parser.parse_args()

    should_record = any((args.record_gif, args.record_best_gif, args.record_success_gif))

    env = make_env(
        seed=123,
        obs_size=args.obs_size,
        domain_randomize=args.randomize_eval,
        obs_camera=args.obs_camera,
        red_zone_xy=(args.red_zone_x, args.red_zone_y),
        blue_zone_xy=(args.blue_zone_x, args.blue_zone_y),
        zone_half=args.zone_half,
        spawn_noise=args.spawn_noise,
        goal_noise=args.goal_noise,
        success_threshold=args.success_threshold,
        park_bonus=args.park_bonus,
        max_steps=args.max_steps,
        record=should_record,
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
    returns, lengths, in_zone_fractions = [], [], []
    recorded = []
    for ep_idx in range(args.n_episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        last_info = {}
        frames = []
        if should_record:
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
            if should_record:
                frame = env.unwrapped.render()
                if frame is not None:
                    frames.append(frame.copy())
        success = bool(last_info.get("is_success", False))
        in_zone_frac = float(last_info.get("in_zone_fraction", 0.0))
        successes += int(success)
        returns.append(total_reward)
        lengths.append(steps)
        in_zone_fractions.append(in_zone_frac)
        if should_record:
            recorded.append({
                "episode": ep_idx + 1,
                "return": total_reward,
                "steps": steps,
                "success": success,
                "in_zone_fraction": in_zone_frac,
                "frames": frames,
            })
        print(f"ep {ep_idx + 1:3d}: return={total_reward:7.2f} steps={steps:3d} "
              f"success={success} in_zone_frac={in_zone_frac:.2f}")

    print()
    print(f"success rate (in-zone at end): {successes}/{args.n_episodes} = {successes / args.n_episodes:.0%}")
    print(f"return mean+-std:             {np.mean(returns):.2f} +- {np.std(returns):.2f}")
    print(f"length mean+-std:             {np.mean(lengths):.1f} +- {np.std(lengths):.1f}")
    print(f"in-zone-fraction mean+-std:    {np.mean(in_zone_fractions):.2f} +- {np.std(in_zone_fractions):.2f}")

    if recorded:
        best = max(recorded, key=lambda ep: float(ep["return"]))
        print(f"best return:    ep {best['episode']}  return={best['return']:.2f}  "
              f"steps={best['steps']}  success={best['success']}  in_zone_frac={best['in_zone_fraction']:.2f}")
        succ = [ep for ep in recorded if bool(ep["success"])]
        if succ:
            best_s = max(succ, key=lambda ep: float(ep["return"]))
            print(f"best success:   ep {best_s['episode']}  return={best_s['return']:.2f}  "
                  f"steps={best_s['steps']}  in_zone_frac={best_s['in_zone_fraction']:.2f}")

    if args.record_gif:
        write_episode_gif(args.record_gif, select_episode_for_gif(recorded, args.record_gif_mode),
                          f"{args.record_gif_mode} evaluated episode")
    if args.record_best_gif:
        write_episode_gif(args.record_best_gif, select_episode_for_gif(recorded, "best"),
                          "best-return evaluated episode")
    if args.record_success_gif:
        write_episode_gif(args.record_success_gif,
                          select_episode_for_gif(recorded, args.record_success_gif_mode),
                          f"{args.record_success_gif_mode} evaluated episode")

    env.close()


if __name__ == "__main__":
    main()
