
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
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from envs.wx250_pick_env_image import WX250PickPlaceImageEnv  # noqa: E402


def make_env(
    domain_randomize,
    seed = 0,
    obs_camera = "front",
    render_mode = None,
):
    def _thunk():
        return WX250PickPlaceImageEnv(
            render_mode=render_mode,
            obs_mode="image",
            domain_randomize=domain_randomize,
            obs_camera=obs_camera,
            seed=seed,
        )

    return _thunk


def _save_gif(frames, out, duration_ms = 50):
    pil_frames = [Image.fromarray(frame) for frame in frames]
    palette_frame = pil_frames[0].convert("P", palette=Image.ADAPTIVE, colors=255, dither=Image.NONE)
    quantized_frames = [palette_frame]
    quantized_frames.extend(frame.quantize(palette=palette_frame, dither=Image.NONE) for frame in pil_frames[1:])
    quantized_frames[0].save(
        out,
        save_all=True,
        append_images=quantized_frames[1:],
        duration=duration_ms,
        loop=0,
    )


def _select_episode_for_gif(episodes, mode):
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


def _write_episode_gif(out_path, selected_episode, label):
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
    _save_gif(frames, out)
    print(f"Saved {len(frames)} frames to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="Path to .zip SAC checkpoint")
    parser.add_argument("--n-episodes", type=int, default=50)
    parser.add_argument("--n-stack", type=int, default=3)
    parser.add_argument("--deterministic", action="store_true", default=True)
    parser.add_argument("--no-randomize", action="store_true")
    parser.add_argument("--obs-camera", choices=WX250PickPlaceImageEnv.OBS_CAMERAS, default="front")
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
    parser.add_argument("--device", default="auto", help="cpu / cuda / auto")
    args = parser.parse_args()
    should_record_episodes = any((args.record_gif, args.record_best_gif, args.record_success_gif))

    env = VecFrameStack(
        DummyVecEnv(
            [
                make_env(
                    domain_randomize=not args.no_randomize,
                    obs_camera=args.obs_camera,
                    render_mode="rgb_array" if should_record_episodes else None,
                )
            ]
        ),
        n_stack=args.n_stack,
    )
    model = SAC.load(args.model, device=args.device)
    print(f"Loaded model from {args.model}   device={model.device}")

    successes = 0
    returns = []
    ep_lens = []
    recorded_episodes = []
    base_env = env.venv.envs[0] if should_record_episodes else None
    for ep in range(args.n_episodes):
        obs = env.reset()
        done = np.array([False])
        episode_return, steps, last_info = 0.0, 0, [{}]
        frames = []
        if base_env is not None:
            frame = base_env.render()
            if frame is not None:
                frames.append(frame.copy())
        while not done.any():
            action, _ = model.predict(obs, deterministic=args.deterministic)
            obs, reward, done, info = env.step(action)
            episode_return += float(reward[0])
            steps += 1
            last_info = info
            if base_env is not None:
                frame = base_env.render()
                if frame is not None:
                    frames.append(frame.copy())
        success = bool(last_info[0].get("is_success", False))
        successes += int(success)
        returns.append(episode_return)
        ep_lens.append(steps)
        if base_env is not None:
            recorded_episodes.append(
                {
                    "episode": ep + 1,
                    "return": episode_return,
                    "steps": steps,
                    "success": success,
                    "frames": frames,
                }
            )
        print(f"  ep {ep + 1:3d}: return={episode_return:7.2f}  steps={steps:3d}  success={success}")

    print()
    print(f"success rate:   {successes}/{args.n_episodes}  =  {successes / args.n_episodes:.0%}")
    print(f"return mean+-std: {np.mean(returns):.2f} +- {np.std(returns):.2f}")
    print(f"length mean+-std: {np.mean(ep_lens):.1f} +- {np.std(ep_lens):.1f}")

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
        selected_episode = _select_episode_for_gif(recorded_episodes, args.record_gif_mode)
        _write_episode_gif(args.record_gif, selected_episode, f"{args.record_gif_mode} evaluated episode")
    if args.record_best_gif:
        best_episode = _select_episode_for_gif(recorded_episodes, "best")
        _write_episode_gif(args.record_best_gif, best_episode, "best-return evaluated episode")
    if args.record_success_gif:
        success_episode = _select_episode_for_gif(recorded_episodes, args.record_success_gif_mode)
        _write_episode_gif(args.record_success_gif, success_episode, f"{args.record_success_gif_mode} evaluated episode")


if __name__ == "__main__":
    main()
