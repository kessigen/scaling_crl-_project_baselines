
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces


PHYSICS_TIMESTEP = 0.002


XML_TEMPLATE = r'''
<mujoco model="wx250_pick_place_image">
  <compiler angle="radian" meshdir="{meshdir}" texturedir="{texturedir}" autolimits="true"/>
  <option timestep="0.002" gravity="0 0 -9.81" integrator="implicitfast" cone="elliptic" impratio="10"/>

  <asset>
    <texture name="interbotix_black" type="2d" file="interbotix_black.png"/>
    <material name="black" texture="interbotix_black"/>

    <texture name="tex_checker"  type="2d" builtin="checker" rgb1="0.24 0.32 0.40" rgb2="0.16 0.22 0.28" width="300" height="300"/>
    <texture name="tex_flat"     type="2d" builtin="flat"    rgb1="0.55 0.45 0.35" width="128" height="128"/>
    <texture name="tex_gradient" type="2d" builtin="gradient" rgb1="0.6 0.5 0.4" rgb2="0.3 0.25 0.2" width="128" height="128"/>

    <material name="table_mat_checker"  texture="tex_checker"  texrepeat="3 3" reflectance="0.05"/>
    <material name="table_mat_flat"     texture="tex_flat"                        reflectance="0.05"/>
    <material name="table_mat_gradient" texture="tex_gradient"                    reflectance="0.05"/>

    <material name="cube_red" rgba="0.85 0.20 0.20 1"/>
    <material name="goal_green" rgba="0.10 0.75 0.35 1"/>

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
      <geom type="sphere" size="0.0045" friction="1.4 0.05 0.01" rgba="0.9 0.1 0.1 0"/>
    </default>
  </default>

  <worldbody>
    <light pos="1.2 0 1.4" dir="-0.6 0 -1" diffuse="0.9 0.9 0.9" specular="0.2 0.2 0.2"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="table_mat_checker"/>
    <geom name="table" type="box" pos="0.33 0 0.02" size="0.26 0.32 0.02" material="table_mat_checker" friction="1.0 0.1 0.02"/>

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
      <site name="cube_site" pos="0 0 0" size="0.010" rgba="1 0 0 0"/>
    </body>

    <body name="goal" pos="{goal_x} {goal_y} 0.058">
      <geom name="goal_marker" type="box" size="{goal_half} {goal_half} 0.001" contype="0" conaffinity="0" material="goal_green"/>
      <site name="goal_site" pos="0 0 0" size="0.010" rgba="0 1 0 0"/>
    </body>

    <camera name="front" pos="0.70 0 0.38" xyaxes="0 1 0 -0.625 0 0.781" fovy="50"/>
    <camera name="isometric" pos="0.65 -0.55 0.55" xyaxes="0.72 0.69 0 -0.42 0.44 0.79" fovy="50"/>
    <camera name="topdown" pos="0.33 0 0.78" xyaxes="1 0 0 0 1 0" fovy="42"/>
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
    pass


class WX250PickPlaceImageEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 20}
    OBS_CAMERAS = ("front", "isometric", "topdown")

    def __init__(
        self,
        render_mode = None,
        frame_skip = 10,
        action_scale = None,
        control_dt = None,
        seed = None,
        asset_root = None,
        obs_mode = "image",
        obs_height = 64,
        obs_width = 64,
        domain_randomize = True,
        obs_camera = "front",
        easy_reset_prob = 0.0,
        easy_distance_scale = 0.5,
        goal_half = 0.05,
        full_goal_success = False,
        legacy_reward = False,
    ):
        super().__init__()
        if obs_mode not in ("image", "pose"):
            raise ValueError(f"obs_mode must be 'image' or 'pose', got {obs_mode!r}")
        if obs_camera not in self.OBS_CAMERAS:
            raise ValueError(f"obs_camera must be one of {self.OBS_CAMERAS}, got {obs_camera!r}")
        if not 0.0 <= easy_reset_prob <= 1.0:
            raise ValueError(f"easy_reset_prob must be in [0, 1], got {easy_reset_prob!r}")
        if not 0.0 < easy_distance_scale <= 1.0:
            raise ValueError(
                f"easy_distance_scale must be in (0, 1], got {easy_distance_scale!r}"
            )
        if goal_half <= 0.0:
            raise ValueError(f"goal_half must be > 0, got {goal_half!r}")
        self.obs_mode = obs_mode
        self.obs_height = obs_height
        self.obs_width = obs_width
        self.domain_randomize = domain_randomize
        self.obs_camera = obs_camera
        self.easy_reset_prob = easy_reset_prob
        self.easy_distance_scale = easy_distance_scale
        self.full_goal_success = full_goal_success
        self.legacy_reward = legacy_reward

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
        self.action_scale = np.array(
            action_scale if action_scale is not None else [0.07, 0.06, 0.06, 0.08, 0.10, 0.005],
            dtype=np.float64,
        )

        # Shrunken inner edge (kinematic singularity + self-occlusion avoidance),
        # slightly extended outer edge to preserve workspace area.
        self.cube_xy_range = np.array([[0.27, 0.42], [-0.13, 0.13]], dtype=np.float64)
        self.goal_xy_range = np.array([[0.23, 0.44], [-0.18, 0.18]], dtype=np.float64)
        self.cube_half_size = 0.018
        self.goal_half = float(goal_half)  # Success half-width and visible green marker half-size.
        if self.full_goal_success and self.goal_half <= self.cube_half_size:
            raise ValueError(
                f"goal_half must be greater than cube half-size ({self.cube_half_size}) "
                "when full_goal_success is enabled."
            )
        self.min_cube_goal_separation = max(0.12, np.sqrt(2.0) * self.goal_half + 0.02)
        self.max_steps = 250
        self.step_count = 0

        # Nominal values used by domain randomization to jitter around.
        self._nominal_cam_pos = {
            "front": np.array([0.70, 0.0, 0.38], dtype=np.float64),
            "isometric": np.array([0.65, -0.55, 0.55], dtype=np.float64),
            "topdown": np.array([0.33, 0.0, 0.78], dtype=np.float64),
        }
        self._nominal_light_pos = np.array([1.2, 0.0, 1.4], dtype=np.float64)

        self.model = None
        self.data = None
        self.renderer = None
        self.obs_renderer = None
        self.viewer = None
        self._xml_temp_path = None

        self._build_model(cube_xy=np.array([0.30, 0.0]), goal_xy=np.array([0.25, -0.12]))

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float32)
        if self.obs_mode == "image":
            self.observation_space = spaces.Box(
                low=0, high=255, shape=(self.obs_height, self.obs_width, 3), dtype=np.uint8
            )
        else:
            obs = self._get_pose_obs()
            self.observation_space = spaces.Box(
                low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32
            )

    @staticmethod
    def _resolve_asset_root(asset_root):
        candidates = []
        if asset_root is not None:
            candidates.append(Path(asset_root))
        here = Path(__file__).resolve().parent
        candidates.extend([
            here / "assets",
            here / "external" / "interbotix_ros_manipulators" / "interbotix_ros_xsarms" / "interbotix_xsarm_descriptions" / "meshes",
            here.parent / "assets",
            here.parent / "assets" / "external" / "interbotix_ros_manipulators" / "interbotix_ros_xsarms" / "interbotix_xsarm_descriptions" / "meshes",
            here.parent / "external" / "interbotix_ros_manipulators" / "interbotix_ros_xsarms" / "interbotix_xsarm_descriptions" / "meshes",
            here.parent.parent / "external" / "interbotix_ros_manipulators" / "interbotix_ros_xsarms" / "interbotix_xsarm_descriptions" / "meshes",
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

    def _build_model(self, cube_xy, goal_xy):
        meshdir = str(self.asset_root).replace('\\', '/')
        texturedir = str(self.asset_root).replace('\\', '/')
        xml = XML_TEMPLATE.format(
            meshdir=meshdir,
            texturedir=texturedir,
            cube_x=float(cube_xy[0]),
            cube_y=float(cube_xy[1]),
            goal_x=float(goal_xy[0]),
            goal_y=float(goal_xy[1]),
            goal_half=float(self.goal_half),
        )
        workspace_tmp = Path(__file__).resolve().parent.parent / ".tmp_mjcf"
        workspace_tmp.mkdir(parents=True, exist_ok=True)
        self._xml_temp_path = workspace_tmp / f"wx250_img_mjcf_{uuid4().hex}.xml"
        self._xml_temp_path.write_text(xml, encoding="utf-8")
        self.model = mujoco.MjModel.from_xml_path(str(self._xml_temp_path))
        self.data = mujoco.MjData(self.model)
        self.renderer = None
        self.obs_renderer = mujoco.Renderer(self.model, self.obs_height, self.obs_width)
        self.viewer = None
        self._cache_ids()
        self._reset_state(cube_xy, goal_xy)

    def _cache_ids(self):
        assert self.model is not None
        m = self.model
        self.grip_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "grip_site")
        self.left_pad_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "left_pad")
        self.right_pad_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "right_pad")
        self.cube_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "cube_site")
        self.goal_site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "goal_site")
        self.cube_free_joint_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "cube_free")
        self.cube_free_qpos_adr = m.jnt_qposadr[self.cube_free_joint_id]
        self.cube_free_dof_adr = m.jnt_dofadr[self.cube_free_joint_id]

        self.cube_mat_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "cube_red")
        self.goal_mat_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, "goal_green")
        self.table_mat_ids = [
            mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, n)
            for n in ("table_mat_checker", "table_mat_flat", "table_mat_gradient")
        ]
        self.table_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "table")
        self.floor_geom_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.obs_cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, self.obs_camera)

    def _sample_xy(self, xy_range):
        return np.array([
            self.np_random.uniform(*xy_range[0]),
            self.np_random.uniform(*xy_range[1]),
        ], dtype=np.float64)

    @staticmethod
    def _clip_xy(xy, xy_range):
        return np.array([
            np.clip(xy[0], *xy_range[0]),
            np.clip(xy[1], *xy_range[1]),
        ], dtype=np.float64)

    def _reset_state(self, cube_xy, goal_xy):
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

    def _randomize_scene(self):
        if not self.domain_randomize:
            return
        assert self.model is not None
        rng = self.np_random
        m = self.model

        # Cube red hue jitter.
        m.mat_rgba[self.cube_mat_id] = [
            rng.uniform(0.70, 0.95),
            rng.uniform(0.10, 0.30),
            rng.uniform(0.10, 0.30),
            1.0,
        ]

        # Goal green hue jitter. Kept clearly green so policy can recognize it.
        m.mat_rgba[self.goal_mat_id] = [
            rng.uniform(0.05, 0.25),
            rng.uniform(0.60, 0.85),
            rng.uniform(0.20, 0.45),
            1.0,
        ]

        # Table material: pattern swap + per-episode tint.
        chosen_mat = int(rng.choice(self.table_mat_ids))
        m.geom_matid[self.table_geom_id] = chosen_mat
        m.geom_matid[self.floor_geom_id] = chosen_mat
        m.mat_rgba[chosen_mat] = [
            rng.uniform(0.3, 1.0),
            rng.uniform(0.3, 1.0),
            rng.uniform(0.3, 1.0),
            1.0,
        ]
        # If checker was picked, also jitter the tile density.
        if chosen_mat == self.table_mat_ids[0]:
            m.mat_texrepeat[chosen_mat] = [rng.uniform(1.5, 6.0), rng.uniform(1.5, 6.0)]

        # Camera pose jitter (position only; orientation kept nominal for sim-to-real simplicity).
        m.cam_pos[self.obs_cam_id] = self._nominal_cam_pos[self.obs_camera] + rng.uniform(-0.03, 0.03, size=3)

        # Light position jitter.
        m.light_pos[0] = self._nominal_light_pos + rng.uniform(-0.25, 0.25, size=3)

    def _get_pose_obs(self):
        assert self.model is not None and self.data is not None
        grip = self.data.site_xpos[self.grip_site_id].copy()
        cube = self.data.site_xpos[self.cube_site_id].copy()
        goal = self.data.site_xpos[self.goal_site_id].copy()
        qpos = self.data.qpos[:6].copy()
        qvel = self.data.qvel[:6].copy()
        cube_vel = self.data.qvel[self.cube_free_dof_adr:self.cube_free_dof_adr + 6].copy()
        return np.concatenate([qpos, qvel, grip, cube, goal, cube - grip, goal - cube]).astype(np.float32)

    def _get_image_obs(self):
        assert self.model is not None and self.data is not None
        assert self.obs_renderer is not None
        self.obs_renderer.update_scene(self.data, camera=self.obs_camera)
        return self.obs_renderer.render().copy()  # (H, W, 3) uint8

    def _get_obs(self):
        if self.obs_mode == "image":
            return self._get_image_obs()
        return self._get_pose_obs()

    def _grasp_detected(self):
        assert self.data is not None
        left = self.data.site_xpos[self.left_pad_id]
        right = self.data.site_xpos[self.right_pad_id]
        cube = self.data.site_xpos[self.cube_site_id]
        pad_mean = 0.5 * (left + right)
        finger_gap = np.linalg.norm(left - right)
        return np.linalg.norm(cube - pad_mean) < 0.03 and finger_gap < 0.06

    def _reward(self):
        assert self.data is not None
        eef = self.data.site_xpos[self.grip_site_id]
        cube = self.data.site_xpos[self.cube_site_id]
        goal = self.data.site_xpos[self.goal_site_id]

        # 3D distance forces the gripper to descend to cube height rather
        # than hovering above it for free XY-aligned reward.
        d_grip_cube = float(np.linalg.norm(eef - cube))
        d_cube_goal = float(np.linalg.norm(cube[:2] - goal[:2]))

        # Square success region matching the visible goal marker. With
        # full_goal_success, the cube center must be far enough inside the
        # marker that the whole cube footprint fits within the square.
        dx = abs(cube[0] - goal[0])
        dy = abs(cube[1] - goal[1])
        success_half = self.goal_half
        if self.full_goal_success:
            success_half = self.goal_half - self.cube_half_size
        success = bool((dx < success_half) and (dy < success_half))

        # Smooth shaping: reaching helps, but pushing only matters when the
        # gripper is actually close enough to influence the cube. Sigma widened
        # from 3 cm to 5 cm to keep the contact gate smooth in 3D.
        reach = 1.0 - np.tanh(8.0 * d_grip_cube)
        push = 1.0 - np.tanh(6.0 * d_cube_goal)
        contact_w = float(np.exp(-((d_grip_cube / 0.05) ** 2)))

        if self.legacy_reward:
            # Original staged reward. Kept for resuming checkpoints trained
            # before the shaping rebalance (stronger hover/parking gradients,
            # smaller success cliff).
            shaped = 0.3 * reach + 0.7 * contact_w * push - 0.02
            success_reward = 100.0
        else:
            shaped = 0.15 * reach + 0.20 * contact_w * push - 0.08
            success_reward = 300.0

        if success:
            reward = success_reward
        else:
            reward = shaped

        breakdown = RewardBreakdown(
            d_grip_cube=d_grip_cube,
            d_cube_goal=d_cube_goal,
            success=100.0 * float(success),
        )
        return reward, breakdown, success

    def reset(self, *, seed = None, options = None):
        super().reset(seed=seed)
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        cube_xy = self._sample_xy(self.cube_xy_range)
        goal_xy = self._sample_xy(self.goal_xy_range)
        if self.easy_reset_prob > 0.0 and self.np_random.random() < self.easy_reset_prob:
            home_xy = np.array([0.30, 0.0], dtype=np.float64)
            scale = self.easy_distance_scale
            cube_xy = home_xy + scale * (cube_xy - home_xy)
            goal_xy = cube_xy + scale * (goal_xy - cube_xy)
            cube_xy = self._clip_xy(cube_xy, self.cube_xy_range)
            goal_xy = self._clip_xy(goal_xy, self.goal_xy_range)
        tries = 0
        while np.linalg.norm(goal_xy - cube_xy) < self.min_cube_goal_separation and tries < 20:
            goal_xy = self._sample_xy(self.goal_xy_range)
            tries += 1
        self._reset_state(cube_xy, goal_xy)
        self._randomize_scene()
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), {"cube_xy": cube_xy, "goal_xy": goal_xy}

    def step(self, action):
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

    def render(self):
        assert self.model is not None and self.data is not None
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data, camera=self.obs_camera)
            return self.renderer.render()
        if self.render_mode == "human":
            if self.viewer is None:
                from mujoco import viewer
                self.viewer = viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
        return None

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.obs_renderer is not None:
            self.obs_renderer.close()
            self.obs_renderer = None
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None


if __name__ == "__main__":
    env = WX250PickPlaceImageEnv(render_mode="rgb_array")
    obs, _ = env.reset(seed=0)
    print("obs shape", obs.shape, "dtype", obs.dtype)
    for _ in range(5):
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        print("step", obs.shape, reward, terminated, truncated, info["is_success"])
    env.close()
