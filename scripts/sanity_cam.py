"""Sanity check: render from several camera configs and measure cube/goal pixel coverage."""

import sys
from pathlib import Path

import numpy as np
import mujoco
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.wx250_pick_env import WX250PickPlaceEnv, XML_TEMPLATE


def count_color_pixels(img, target_rgb, tol=40):
    diff = np.abs(img.astype(np.int16) - np.array(target_rgb, dtype=np.int16))
    mask = np.all(diff < tol, axis=-1)
    return int(mask.sum()), mask


def render(model, data, cam_name, h, w):
    r = mujoco.Renderer(model, h, w)
    r.update_scene(data, camera=cam_name)
    img = r.render().copy()
    r.close()
    return img


def main():
    env = WX250PickPlaceEnv()
    env.reset(seed=0)

    # Place cube and goal at representative positions for measurement.
    cube_xy = np.array([0.30, 0.00])
    goal_xy = np.array([0.25, 0.15])
    env._reset_state(cube_xy, goal_xy)
    mujoco.mj_forward(env.model, env.data)

    print(f"obs dtype={env._get_obs().dtype}, shape={env._get_obs().shape}")
    print(f"obs_space={env.observation_space}")
    print(f"action_space={env.action_space}")

    cube_pos = env.data.site_xpos[env.cube_site_id]
    goal_pos = env.data.site_xpos[env.goal_site_id]
    grip_pos = env.data.site_xpos[env.grip_site_id]
    print(f"cube world pos: {cube_pos}")
    print(f"goal world pos: {goal_pos}")
    print(f"grip world pos: {grip_pos}")

    # Render at 480x640 from existing front camera
    img_big = render(env.model, env.data, "front", 480, 640)
    Image.fromarray(img_big).save("sanity_front_480.png")

    # Render at 84x84 from existing front camera (what CNN would see)
    img_84 = render(env.model, env.data, "front", 84, 84)
    Image.fromarray(img_84).save("sanity_front_84.png")

    # Pixel coverage at 84x84 of cube (red) and goal (green)
    cube_px_84, _ = count_color_pixels(img_84, (217, 51, 51))
    goal_px_84, _ = count_color_pixels(img_84, (25, 191, 89))
    print(f"\n[current front cam @ 84x84]  cube pixels: {cube_px_84}   goal pixels: {goal_px_84}")

    # Also 64x64
    img_64 = render(env.model, env.data, "front", 64, 64)
    Image.fromarray(img_64).save("sanity_front_64.png")
    cube_px_64, _ = count_color_pixels(img_64, (217, 51, 51))
    goal_px_64, _ = count_color_pixels(img_64, (25, 191, 89))
    print(f"[current front cam @ 64x64]  cube pixels: {cube_px_64}   goal pixels: {goal_px_64}")

    # Now try proposed new camera position (0.70, 0, 0.38) with 50deg FOV.
    # Edit XML in place on the existing model is not trivial; easier to rebuild model.
    cam_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_CAMERA, "front")
    env.model.cam_pos[cam_id] = np.array([0.70, 0.0, 0.38])
    # xyaxes = right_axis(0,1,0), up_axis(-0.42,0,0.91) -> fovy unchanged from default 45
    # we only change pos/quat; approximate by setting the camera look-at direction.
    # For a quick check, just change pos and FOV.
    env.model.cam_fovy[cam_id] = 50.0
    mujoco.mj_forward(env.model, env.data)

    img_new_84 = render(env.model, env.data, "front", 84, 84)
    Image.fromarray(img_new_84).save("sanity_front_proposed_84.png")
    img_new_big = render(env.model, env.data, "front", 480, 640)
    Image.fromarray(img_new_big).save("sanity_front_proposed_480.png")
    cube_px_new, _ = count_color_pixels(img_new_84, (217, 51, 51))
    goal_px_new, _ = count_color_pixels(img_new_84, (25, 191, 89))
    print(f"[proposed pos(0.70,0,0.38) FOV50 @ 84x84]  cube pixels: {cube_px_new}   goal pixels: {goal_px_new}")

    # Render isometric and top for reference
    img_iso = render(env.model, env.data, "isometric", 84, 84)
    Image.fromarray(img_iso).save("sanity_iso_84.png")

    # Also try a top-down camera for comparison - add a synthetic camera pose
    # Reuse cam_id=front with a top position
    env.model.cam_pos[cam_id] = np.array([0.30, 0.0, 0.80])
    # Note: we can't easily change orientation here without quaternion; fovy stays 50
    # The existing orientation still points roughly horizontally, so this won't be
    # a true top-down, but we can see pixel scaling
    img_top_like = render(env.model, env.data, "front", 84, 84)
    Image.fromarray(img_top_like).save("sanity_hi_84.png")

    # Verify VecFrameStack shape
    from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
    venv = VecFrameStack(DummyVecEnv([lambda: WX250PickPlaceEnv()]), n_stack=3)
    stacked_obs = venv.reset()
    print(f"\n[VecFrameStack n_stack=3] obs shape={stacked_obs.shape}, dtype={stacked_obs.dtype}")

    print("\nPNGs written to working dir: sanity_front_{480,84,64,proposed_480,proposed_84}.png, sanity_iso_84.png, sanity_hi_84.png")


if __name__ == "__main__":
    main()
