from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import tempfile

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


PHYSICS_TIMESTEP = 0.002


XML_TEMPLATE = r'''
<mujoco model="wx250_pick_place_mesh">
  <compiler angle="radian" meshdir="{meshdir}" texturedir="{texturedir}" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="10"/>

  <asset>
    <texture name="interbotix_black" type="2d" file="interbotix_black.png"/>
    <material name="black" texture="interbotix_black"/>
    <texture name="grid" type="2d" builtin="checker" rgb1="0.24 0.32 0.40" rgb2="0.16 0.22 0.28" width="300" height="300"/>
    <material name="table_mat" texture="grid" texrepeat="3 3" reflectance="0.05"/>
    <material name="cube_red" rgba="0.85 0.20 0.20 1"/>
    <material name="goal_green" rgba="0.10 0.75 0.35 0.35"/>

    <mesh name="wx250_1_base" file="wx250_meshes/wx250_1_base.stl" scale="0.001 0.001 0.001"/>
    <mesh name="wx250_2_shoulder" file="wx250_meshes/wx250_2_shoulder.stl" scale="0.001 0.001 0.001"/>
    <mesh name="wx250_3_upper_arm" file="wx250_meshes/wx250_3_upper_arm.stl" scale="0.001 0.001 0.001"/>
    <mesh name="wx250_4_forearm" file="wx250_meshes/wx250_4_forearm.stl" scale="0.001 0.001 0.001"/>
    <mesh name="wx250_5_wrist" file="wx250_meshes/wx250_5_wrist.stl" scale="0.001 0.001 0.001"/>
    <mesh name="wx250_6_gripper" file="wx250_meshes/wx250_6_gripper.stl" scale="0.001 0.001 0.001"/>
    <mesh name="wx250_8_gripper_bar" file="wx250_meshes/wx250_8_gripper_bar.stl" scale="0.001 0.001 0.001"/>
    <mesh name="wx250_9_gripper_finger" file="wx250_meshes/wx250_9_gripper_finger.stl" scale="0.001 0.001 0.001"/>
  </asset>

  <default>
    <default class="arm_joint">
      <joint damping="1.5" frictionloss="0.1" armature="0.05"/>
      <position kp="45" dampratio="1.0" forcerange="-18 18"/>
    </default>
    <default class="finger_joint">
      <joint damping="3.0" frictionloss="0.1" armature="0.01"/>
      <position kp="180" dampratio="1.0" forcerange="-8 8"/>
    </default>
    <default class="visual">
      <geom type="mesh" contype="0" conaffinity="0" group="2" material="black" density="0"/>
    </default>
    <default class="collision">
      <geom type="mesh" group="3" friction="1.0 0.05 0.01" margin="0.002"/>
    </default>
    <default class="pad_collision">
      <geom type="sphere" size="0.0045" friction="1.4 0.05 0.01" rgba="0.9 0.1 0.1 1"/>
    </default>
  </default>

  <worldbody>
    <light pos="1.2 0 1.4" dir="-0.6 0 -1" diffuse="0.9 0.9 0.9" specular="0.2 0.2 0.2"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="table_mat"/>
    <geom name="table" type="box" pos="0.33 0 0.02" size="0.26 0.32 0.02" material="table_mat" friction="1.0 0.1 0.02"/>

    <body name="wx250/base_link" pos="0 0 0.04">
      <inertial pos="-0.0380446 0.000613892 0.0193354" quat="0.509292 0.490887 -0.496359 0.503269" mass="0.538736"
        diaginertia="0.00252518 0.00211519 0.000690737"/>
      <geom quat="1 0 0 1" mesh="wx250_1_base" class="visual"/>
      <geom quat="1 0 0 1" mesh="wx250_1_base" class="collision"/>

      <body name="wx250/shoulder_link" pos="0 0 0.072">
        <inertial pos="2.23482e-05 4.14609e-05 0.0066287" quat="0.0130352 0.706387 0.012996 0.707586" mass="0.480879"
          diaginertia="0.000588946 0.000555655 0.000378999"/>
        <joint name="waist" class="arm_joint" axis="0 0 1" range="-3.14158 3.14158"/>
        <geom pos="0 0 -0.003" quat="1 0 0 1" mesh="wx250_2_shoulder" class="visual"/>
        <geom pos="0 0 -0.003" quat="1 0 0 1" mesh="wx250_2_shoulder" class="collision"/>

        <body name="wx250/upper_arm_link" pos="0 0 0.03865">
          <inertial pos="0.0171605 2.725e-07 0.191323" quat="0.705539 0.0470667 -0.0470667 0.705539" mass="0.430811"
            diaginertia="0.00364425 0.003463 0.000399348"/>
          <joint name="shoulder" class="arm_joint" axis="0 1 0" range="-1.88496 1.98968"/>
          <geom quat="1 0 0 1" mesh="wx250_3_upper_arm" class="visual"/>
          <geom quat="1 0 0 1" mesh="wx250_3_upper_arm" class="collision"/>

          <body name="wx250/forearm_link" pos="0.04975 0 0.25">
            <inertial pos="0.153423 -0.0001179685 -0.000439" quat="0.706899 0.0181438 -0.0181438 0.706849" mass="0.297673"
              diaginertia="0.00218601 0.00217599 5.75698e-05"/>
            <joint name="elbow" class="arm_joint" axis="0 1 0" range="-2.14675 1.60570"/>
            <geom quat="1 0 0 1" mesh="wx250_4_forearm" class="visual"/>
            <geom quat="1 0 0 1" mesh="wx250_4_forearm" class="collision"/>

            <body name="wx250/wrist_link" pos="0.25 0 0">
              <inertial pos="0.04236 -1.0663e-05 0.010577" quat="0.608721 0.363497 -0.359175 0.606895" mass="0.084957"
                diaginertia="3.29057e-05 3.082e-05 2.68343e-05"/>
              <joint name="wrist_angle" class="arm_joint" axis="0 1 0" range="-1.74533 2.14675"/>
              <geom quat="1 0 0 1" mesh="wx250_5_wrist" class="visual"/>
              <geom quat="1 0 0 1" mesh="wx250_5_wrist" class="collision"/>

              <body name="wx250/gripper_link" pos="0.065 0 0">
                <inertial pos="0.021631 2.516e-07 0.01141" quat="0.708234 0.0260737 -0.0260737 0.705014" mass="0.072885"
                  diaginertia="2.537e-05 1.836e-05 1.674e-05"/>
                <joint name="wrist_rotate" class="arm_joint" axis="1 0 0" range="-3.14158 3.14158"/>
                <geom pos="-0.02 0 0" quat="1 0 0 1" mesh="wx250_6_gripper" class="visual"/>
                <geom pos="-0.02 0 0" quat="1 0 0 1" mesh="wx250_6_gripper" class="collision"/>
                <geom pos="-0.02 0 0" quat="1 0 0 1" mesh="wx250_8_gripper_bar" class="visual"/>
                <geom pos="-0.02 0 0" quat="1 0 0 1" mesh="wx250_8_gripper_bar" class="collision"/>
                <site name="grip_site" pos="0.093 0 0" size="0.009" rgba="0 0.6 1 0"/>

                <body name="left_finger_link" pos="0.093 0 0">
                  <inertial pos="0.013816 0 0" quat="0.705384 0.705384 -0.0493271 -0.0493271" mass="0.016246"
                    diaginertia="4.79509e-06 3.7467e-06 1.48651e-06"/>
                  <joint name="left_finger" class="finger_joint" axis="0 1 0" type="slide" range="0.015 0.037"/>
                  <geom pos="0 0.005 0" quat="0 0 0 -1" mesh="wx250_9_gripper_finger" class="visual"/>
                  <geom pos="0 0.005 0" quat="0 0 0 -1" mesh="wx250_9_gripper_finger" class="collision"/>
                  <geom name="left_pad0" pos="0.042 -0.009 0.012" class="pad_collision"/>
                  <geom name="left_pad1" pos="0.042 -0.009 -0.012" class="pad_collision"/>
                  <site name="left_pad" pos="0.05 -0.008 0" size="0.004" rgba="1 0 0 0"/>
                </body>

                <body name="right_finger_link" pos="0.093 0 0">
                  <inertial pos="0.013816 0 0" quat="0.705384 0.705384 0.0493271 0.0493271" mass="0.016246"
                    diaginertia="4.79509e-06 3.7467e-06 1.48651e-06"/>
                  <joint name="right_finger" class="finger_joint" axis="0 1 0" type="slide" range="-0.037 -0.015"/>
                  <geom pos="0 -0.005 0" quat="0 0 1 0" mesh="wx250_9_gripper_finger" class="visual"/>
                  <geom pos="0 -0.005 0" quat="0 0 1 0" mesh="wx250_9_gripper_finger" class="collision"/>
                  <geom name="right_pad0" pos="0.042 0.009 0.012" class="pad_collision"/>
                  <geom name="right_pad1" pos="0.042 0.009 -0.012" class="pad_collision"/>
                  <site name="right_pad" pos="0.05 0.008 0" size="0.004" rgba="1 0 0 0"/>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <body name="cube" pos="{cube_x} {cube_y} 0.06">
      <freejoint name="cube_free"/>
      <geom name="cube_geom" type="box" size="0.018 0.018 0.018" mass="0.045" material="cube_red" friction="1.1 0.1 0.02"/>
      <site name="cube_site" pos="0 0 0" size="0.010" rgba="1 0 0 1"/>
    </body>

    <body name="goal" pos="{goal_x} {goal_y} 0.058">
      <geom name="goal_marker" type="cylinder" size="0.025 0.0015" contype="0" conaffinity="0" material="goal_green"/>
      <site name="goal_site" pos="0 0 0" size="0.010" rgba="0 1 0 1"/>
    </body>

    <camera name="front" pos="0.85 0 0.55" xyaxes="0 1 0 -0.35 0 0.94"/>
    <camera name="isometric" pos="0.65 -0.55 0.55" xyaxes="0.72 0.69 0 -0.42 0.44 0.79"/>
  </worldbody>

  <contact>
    <exclude body1="wx250/base_link" body2="wx250/shoulder_link"/>
    <exclude body1="wx250/shoulder_link" body2="wx250/upper_arm_link"/>
    <exclude body1="wx250/upper_arm_link" body2="wx250/forearm_link"/>
    <exclude body1="wx250/forearm_link" body2="wx250/wrist_link"/>
    <exclude body1="wx250/wrist_link" body2="wx250/gripper_link"/>
    <exclude body1="wx250/gripper_link" body2="left_finger_link"/>
    <exclude body1="wx250/gripper_link" body2="right_finger_link"/>
  </contact>

  <equality>
    <joint joint1="left_finger" joint2="right_finger" polycoef="0 -1 0 0 0"/>
  </equality>

  <actuator>
    <position class="arm_joint" name="waist_act" joint="waist"/>
    <position class="arm_joint" name="shoulder_act" joint="shoulder"/>
    <position class="arm_joint" name="elbow_act" joint="elbow"/>
    <position class="arm_joint" name="wrist_angle_act" joint="wrist_angle"/>
    <position class="arm_joint" name="wrist_rotate_act" joint="wrist_rotate"/>
    <position class="finger_joint" name="gripper_act" joint="left_finger"/>
  </actuator>

  <keyframe>
    <key name="home" qpos="0 -0.95 1.10 -0.35 0 0.022 -0.022 0.30 0 0.06 1 0 0 0" ctrl="0 -0.95 1.10 -0.35 0 0.022"/>
  </keyframe>
</mujoco>
'''


@dataclass
class RewardBreakdown:
    d_grip_cube: float
    d_cube_goal: float
    success: float


class WX250PickPlaceEnv(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}

    def __init__(
        self,
        render_mode: str | None = None,
        frame_skip: int = 10,
        action_scale: np.ndarray | None = None,
        control_dt: float | None = None,
        seed: int | None = None,
        asset_root: str | Path | None = None,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        if frame_skip < 1:
            raise ValueError("frame_skip must be >= 1")
        if control_dt is None:
            self.frame_skip = frame_skip
            self.control_dt = PHYSICS_TIMESTEP * self.frame_skip
        else:
            if control_dt <= 0:
                raise ValueError("control_dt must be > 0")
            derived_frame_skip = int(round(control_dt / PHYSICS_TIMESTEP))
            if derived_frame_skip < 1 or not np.isclose(
                derived_frame_skip * PHYSICS_TIMESTEP,
                control_dt,
                atol=1e-9,
            ):
                raise ValueError(
                    f"control_dt must be a positive multiple of the physics timestep ({PHYSICS_TIMESTEP})."
                )
            self.frame_skip = derived_frame_skip
            self.control_dt = PHYSICS_TIMESTEP * self.frame_skip
        self.np_random = np.random.default_rng(seed)
        self.asset_root = self._resolve_asset_root(asset_root)

        self.home_qpos = np.array([0.0, -0.95, 1.10, -0.35, 0.0, 0.022, -0.022], dtype=np.float64)
        self.ctrl_home = np.array([0.0, -0.95, 1.10, -0.35, 0.0, 0.022], dtype=np.float64)
        self.arm_low = np.array([-3.14158, -1.88496, -2.14675, -1.74533, -3.14158], dtype=np.float64)
        self.arm_high = np.array([3.14158, 1.98968, 1.60570, 2.14675, 3.14158], dtype=np.float64)
        self.grip_low = np.array([0.015], dtype=np.float64)
        self.grip_high = np.array([0.037], dtype=np.float64)
        self.ctrl_low = np.concatenate([self.arm_low, self.grip_low])
        self.ctrl_high = np.concatenate([self.arm_high, self.grip_high])
        self.action_scale = np.array(action_scale if action_scale is not None else [0.07, 0.06, 0.06, 0.08, 0.10, 0.005], dtype=np.float64)

        self.cube_xy_range = np.array([[0.22, 0.39], [-0.13, 0.13]], dtype=np.float64)
        self.goal_xy_range = np.array([[0.18, 0.40], [-0.18, 0.18]], dtype=np.float64)
        self.success_threshold = 0.04
        self.max_steps = 250
        self.step_count = 0

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None
        self.renderer: mujoco.Renderer | None = None
        self.viewer = None
        self._xml_temp_path: Path | None = None

        self._build_model(cube_xy=np.array([0.30, 0.0]), goal_xy=np.array([0.25, -0.12]))

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        obs = self._get_obs()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32)

    @staticmethod
    def _resolve_asset_root(asset_root: str | Path | None) -> Path:
        candidates: list[Path] = []
        if asset_root is not None:
            candidates.append(Path(asset_root))
        here = Path(__file__).resolve().parent
        candidates.extend([
            here / "assets",
            here / "external" / "interbotix_ros_manipulators" / "interbotix_ros_xsarms" / "interbotix_xsarm_descriptions" / "meshes",
            here.parent / "external" / "interbotix_ros_manipulators" / "interbotix_ros_xsarms" / "interbotix_xsarm_descriptions" / "meshes",
        ])
        for cand in candidates:
            if (cand / "interbotix_black.png").exists() and (cand / "wx250_meshes" / "wx250_1_base.stl").exists():
                return cand
        tried = "\n  - ".join(str(c) for c in candidates)
        raise FileNotFoundError(
            "Could not find Interbotix mesh assets. Expected a directory containing "
            "interbotix_black.png and wx250_meshes/*.stl. Tried:\n  - " + tried +
            "\nRun setup_interbotix_assets.py or pass asset_root=..."
        )

    def _build_model(self, cube_xy: np.ndarray, goal_xy: np.ndarray) -> None:
        meshdir = str(self.asset_root).replace('\\', '/')
        texturedir = str(self.asset_root).replace('\\', '/')
        xml = XML_TEMPLATE.format(meshdir=meshdir, texturedir=texturedir, cube_x=float(cube_xy[0]), cube_y=float(cube_xy[1]), goal_x=float(goal_xy[0]), goal_y=float(goal_xy[1]))
        tmpdir = Path(tempfile.mkdtemp(prefix="wx250_mjcf_"))
        self._xml_temp_path = tmpdir / "scene.xml"
        self._xml_temp_path.write_text(xml, encoding="utf-8")
        self.model = mujoco.MjModel.from_xml_path(str(self._xml_temp_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.viewer = None
        self._cache_ids()
        self._reset_state(cube_xy, goal_xy)

    def _cache_ids(self) -> None:
        assert self.model is not None
        self.grip_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        self.left_pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_pad")
        self.right_pad_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_pad")
        self.cube_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "cube_site")
        self.goal_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "goal_site")
        self.cube_free_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self.cube_free_qpos_adr = self.model.jnt_qposadr[self.cube_free_joint_id]
        self.cube_free_dof_adr = self.model.jnt_dofadr[self.cube_free_joint_id]

    def _sample_xy(self, xy_range: np.ndarray) -> np.ndarray:
        return np.array([
            self.np_random.uniform(*xy_range[0]),
            self.np_random.uniform(*xy_range[1]),
        ], dtype=np.float64)

    def _reset_state(self, cube_xy: np.ndarray, goal_xy: np.ndarray) -> None:
        assert self.model is not None and self.data is not None
        self.data.qpos[:7] = self.home_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = self.ctrl_home
        cube_qpos = np.array([cube_xy[0], cube_xy[1], 0.06, 1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qpos[self.cube_free_qpos_adr:self.cube_free_qpos_adr + 7] = cube_qpos
        goal_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "goal")
        self.model.body_pos[goal_body_id] = np.array([goal_xy[0], goal_xy[1], 0.058], dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        self.step_count = 0

    def _get_obs(self) -> np.ndarray:
        assert self.model is not None and self.data is not None
        grip = self.data.site_xpos[self.grip_site_id].copy()
        cube = self.data.site_xpos[self.cube_site_id].copy()
        goal = self.data.site_xpos[self.goal_site_id].copy()
        qpos = self.data.qpos[:6].copy()
        qvel = self.data.qvel[:6].copy()
        cube_vel = self.data.qvel[self.cube_free_dof_adr:self.cube_free_dof_adr + 6].copy()
        return np.concatenate([qpos, qvel, grip, cube, goal, cube - grip, goal - cube]).astype(np.float32)

    def _grasp_detected(self) -> bool:
        assert self.data is not None
        left = self.data.site_xpos[self.left_pad_id]
        right = self.data.site_xpos[self.right_pad_id]
        cube = self.data.site_xpos[self.cube_site_id]
        pad_mean = 0.5 * (left + right)
        finger_gap = np.linalg.norm(left - right)
        return np.linalg.norm(cube - pad_mean) < 0.03 and finger_gap < 0.06

    def _reward(self) -> tuple[float, RewardBreakdown, bool]:
        assert self.data is not None
        eef = self.data.site_xpos[self.grip_site_id]
        cube = self.data.site_xpos[self.cube_site_id]
        goal = self.data.site_xpos[self.goal_site_id]

        d_grip_cube = float(np.linalg.norm(eef - cube))
        d_cube_goal = float(np.linalg.norm(cube - goal))
        success = d_cube_goal < self.success_threshold
        reward = -0.5 * d_grip_cube - d_cube_goal + 5.0 * float(success)

        breakdown = RewardBreakdown(d_grip_cube=d_grip_cube, d_cube_goal=d_cube_goal, success=5.0 * float(success))
        return reward, breakdown, success

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        cube_xy = self._sample_xy(self.cube_xy_range)
        goal_xy = self._sample_xy(self.goal_xy_range)
        while np.linalg.norm(goal_xy - cube_xy) < 0.08:
            goal_xy = self._sample_xy(self.goal_xy_range)
        self._reset_state(cube_xy, goal_xy)
        return self._get_obs(), {"cube_xy": cube_xy, "goal_xy": goal_xy}

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        assert self.model is not None and self.data is not None
        action = np.asarray(action, dtype=np.float64)
        delta = np.clip(action, -1.0, 1.0) * self.action_scale
        target = np.clip(self.data.ctrl + delta, self.ctrl_low, self.ctrl_high)
        self.data.ctrl[:] = target
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.step_count += 1

        reward, breakdown, success = self._reward()
        obs = self._get_obs()
        terminated = success
        truncated = self.step_count >= self.max_steps
        info = {"reward_breakdown": breakdown.__dict__, "is_success": success}
        if self.render_mode == "human":
            self.render()
        return obs, float(reward), terminated, truncated, info

    def render(self) -> np.ndarray | None:
        assert self.model is not None and self.data is not None
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, camera="isometric")
            return self.renderer.render()
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        return None

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    env = WX250PickPlaceEnv(render_mode="rgb_array")
    obs, _ = env.reset()
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        frame = env.render()
        print(obs.shape, reward, terminated, truncated, frame.shape if frame is not None else None)
    env.close()
