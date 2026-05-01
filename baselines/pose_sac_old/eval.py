
import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from stable_baselines3 import SAC

from envs.wx250_pick_env import WX250PickPlaceEnv


def main():
    parser = argparse.ArgumentParser()
    default_asset_root = str(REPO_ROOT / "assets" / "external" / "interbotix_ros_manipulators" / "interbotix_ros_xsarms" / "interbotix_xsarm_descriptions" / "meshes")
    parser.add_argument("--asset-root", default=default_asset_root)
    parser.add_argument("--model", default="runs/wx250_pose_sac/best_model/best_model.zip")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=None, help="Override episode length (default uses env's max_steps=250)")
    args = parser.parse_args()

    model_path = Path(args.model)
    env = WX250PickPlaceEnv(render_mode="human", asset_root=args.asset_root)
    if args.max_steps is not None:
        env.max_steps = args.max_steps
    model = SAC.load(model_path, env=env)

    for ep in range(args.episodes):
        obs, info = env.reset()
        done = False
        trunc = False
        total_reward = 0.0
        while not (done or trunc):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, trunc, info = env.step(action)
            total_reward += reward
            time.sleep(env.control_dt)
        print(f"episode={ep} total_reward={total_reward:.3f} success={info.get('is_success', False)}")
        time.sleep(1.0)

    print("Press Enter to close...")
    input()
    env.close()


if __name__ == "__main__":
    main()
