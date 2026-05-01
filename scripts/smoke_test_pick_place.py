
import sys
import time
from pathlib import Path

import numpy as np
import mujoco

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.wx250_pick_env import WX250PickPlaceEnv


ASSET_ROOT = REPO_ROOT / "assets" / "external" / "interbotix_ros_manipulators" / "interbotix_ros_xsarms" / "interbotix_xsarm_descriptions" / "meshes"


def get_positions(env):
    grip = env.data.site_xpos[env.grip_site_id].copy()
    cube = env.data.site_xpos[env.cube_site_id].copy()
    goal = env.data.site_xpos[env.goal_site_id].copy()
    return grip, cube, goal


def arm_action_from_target(
    env,
    target_pos,
    grip_cmd = 0.0,
    pos_gain = 8.0,
    damping = 1e-3,
):
    """
    Compute a 6D env action:
      first 5 dims = joint target deltas for arm joints
      last dim     = gripper delta
    using damped least squares on the grip_site position Jacobian.
    """
    grip_pos = env.data.site_xpos[env.grip_site_id].copy()
    err = target_pos - grip_pos

    jacp = np.zeros((3, env.model.nv), dtype=np.float64)
    jacr = np.zeros((3, env.model.nv), dtype=np.float64)
    mujoco.mj_jacSite(env.model, env.data, jacp, jacr, env.grip_site_id)

    # First 5 DoFs are the arm joints in this env:
    # waist, shoulder, elbow, wrist_angle, wrist_rotate
    J = jacp[:, :5]

    # Damped least squares
    A = J @ J.T + damping * np.eye(3)
    dq = J.T @ np.linalg.solve(A, pos_gain * err)

    action = np.zeros(6, dtype=np.float32)
    # Convert desired joint delta to normalized env action
    action[:5] = dq / env.action_scale[:5]
    action[5] = grip_cmd / env.action_scale[5]
    action = np.clip(action, -1.0, 1.0)
    return action


def run_phase(
    env,
    target_pos,
    grip_cmd,
    steps,
    tol,
    sleep_s = 0.01,
    name = "",
):
    for i in range(steps):
        action = arm_action_from_target(env, target_pos, grip_cmd=grip_cmd)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        grip_pos = env.data.site_xpos[env.grip_site_id].copy()
        err = np.linalg.norm(target_pos - grip_pos)

        if i % 20 == 0:
            print(f"[{name}] step={i:03d} err={err:.4f} reward={reward:.3f} success={info['is_success']}")

        if err < tol:
            break

        if terminated or truncated:
            return terminated, truncated, info

        time.sleep(sleep_s)

    return False, False, {"is_success": False}


def run_hold(
    env,
    target_pos,
    grip_cmd,
    steps,
    sleep_s = 0.01,
    name = "",
):
    for i in range(steps):
        action = arm_action_from_target(env, target_pos, grip_cmd=grip_cmd)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()

        if i % 20 == 0:
            print(f"[{name}] step={i:03d} reward={reward:.3f} success={info['is_success']}")

        if terminated or truncated:
            return terminated, truncated, info

        time.sleep(sleep_s)

    return False, False, {"is_success": False}


def one_episode(env):
    obs, info = env.reset()
    env.render()
    time.sleep(0.5)

    grip, cube, goal = get_positions(env)

    cube_top = cube + np.array([0.0, 0.0, 0.045])
    cube_grasp = cube + np.array([0.0, 0.0, 0.015])
    cube_lift = cube + np.array([0.0, 0.0, 0.12])

    goal_top = goal + np.array([0.0, 0.0, 0.12])
    goal_place = goal + np.array([0.0, 0.0, 0.05])

    print("cube:", cube)
    print("goal:", goal)

    # 1) open gripper and move above cube
    terminated, truncated, info = run_phase(
        env, cube_top, grip_cmd=+0.02, steps=120, tol=0.02, name="move_above_cube"
    )
    if terminated or truncated:
        return bool(info.get("is_success", False))

    # 2) descend toward grasp
    terminated, truncated, info = run_phase(
        env, cube_grasp, grip_cmd=+0.01, steps=120, tol=0.015, name="descend_to_cube"
    )
    if terminated or truncated:
        return bool(info.get("is_success", False))

    # 3) close gripper in place
    terminated, truncated, info = run_hold(
        env, cube_grasp, grip_cmd=-0.02, steps=80, name="close_gripper"
    )
    if terminated or truncated:
        return bool(info.get("is_success", False))

    # 4) lift
    terminated, truncated, info = run_phase(
        env, cube_lift, grip_cmd=-0.01, steps=140, tol=0.02, name="lift_cube"
    )
    if terminated or truncated:
        return bool(info.get("is_success", False))

    # 5) move above goal
    terminated, truncated, info = run_phase(
        env, goal_top, grip_cmd=-0.01, steps=180, tol=0.025, name="move_above_goal"
    )
    if terminated or truncated:
        return bool(info.get("is_success", False))

    # 6) descend to place
    terminated, truncated, info = run_phase(
        env, goal_place, grip_cmd=-0.005, steps=120, tol=0.02, name="descend_to_goal"
    )
    if terminated or truncated:
        return bool(info.get("is_success", False))

    # 7) open gripper to release
    terminated, truncated, info = run_hold(
        env, goal_place, grip_cmd=+0.02, steps=80, name="release"
    )
    if terminated or truncated:
        return bool(info.get("is_success", False))

    # 8) retreat
    terminated, truncated, info = run_phase(
        env, goal_top, grip_cmd=+0.02, steps=100, tol=0.03, name="retreat"
    )

    final_success = bool(info.get("is_success", False))
    _, cube_final, goal_final = get_positions(env)
    place_dist = np.linalg.norm(cube_final - goal_final)

    print("final cube:", cube_final)
    print("final goal:", goal_final)
    print("final place dist:", place_dist)
    print("final success:", final_success)
    return final_success


def main():
    env = WX250PickPlaceEnv(
        render_mode="human",
        asset_root=ASSET_ROOT,
    )

    try:
        success = one_episode(env)
        print("\nSMOKE TEST RESULT:", "PASS" if success else "NOT YET SUCCESS")
        print("If it moves sensibly but does not complete, the env is still rendering/stepping correctly.")
    finally:
        env.close()


if __name__ == "__main__":
    main()
