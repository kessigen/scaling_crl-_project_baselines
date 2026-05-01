# WX250 Pick-and-Place with SAC

A Gymnasium + MuJoCo environment for training a 5-DOF Interbotix WX250 robotic arm to perform cube pick-and-place tasks, using Soft Actor-Critic (SAC) via Stable-Baselines3.

The environment uses the actual Interbotix WX250 STL mesh assets for both visual and collision geometry, with a MuJoCo simulation scene built around the robot.

## Prerequisites

- Python 3.9+
- Git (for cloning the Interbotix mesh assets)
- A [Weights & Biases](https://wandb.ai) account (for training logging)

## Setup

### 1. Clone the Interbotix mesh assets

```bash
python setup_interbotix_assets.py
```

This shallow-clones the public Interbotix repo into `external/interbotix_ros_manipulators/`. You can specify a different location with `--target`:

```bash
python setup_interbotix_assets.py --target /path/to/interbotix_ros_manipulators
```

### 2. Install Python dependencies

```bash
pip install gymnasium mujoco stable-baselines3[extra] numpy wandb
```

### 3. Set your Weights & Biases API key

The training script requires a `WANDB_API_KEY` environment variable.

**Linux / macOS:**
```bash
export WANDB_API_KEY="your_key_here"
```

**Windows (PowerShell):**
```powershell
$env:WANDB_API_KEY = "your_key_here"
```

To persist across sessions, add it to your shell profile or set it as a system/user environment variable.

You can find your API key at [https://wandb.ai/authorize](https://wandb.ai/authorize).

## Usage

### Training

```bash
python train_sb3_sac.py --timesteps 500000
```

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--asset-root` | `external/.../meshes` | Path to the Interbotix mesh directory |
| `--timesteps` | `500000` | Total training timesteps |
| `--run-dir` | `runs/wx250_sac_mesh` | Directory for checkpoints, eval logs, and the final model |
| `--drive-dir` | `None` | Optional Google Drive root for persistent Colab outputs |
| `--resume` | off | Resume training from the latest checkpoint in `--run-dir` |

**Outputs** (saved to `--run-dir`):
- `checkpoints/` -- periodic model and replay buffer snapshots
- `best_model/` -- best model by evaluation reward
- `eval_logs/` -- evaluation metrics
- `final_model.zip` -- model at end of training
- `final_replay_buffer.pkl` -- replay buffer at end of training

Training metrics are logged to the `wx250-pick-place` W&B project.

### Evaluating a trained policy

```bash
python eval_policy.py --model runs/wx250_sac_mesh/best_model/best_model.zip
```

This opens a MuJoCo viewer and runs the policy for 5 episodes.

**Arguments:**

| Flag | Default | Description |
|---|---|---|
| `--asset-root` | `external/.../meshes` | Path to the Interbotix mesh directory |
| `--model` | `runs/wx250_sac_mesh/best_model/best_model.zip` | Path to a saved SAC model |
| `--episodes` | `5` | Number of evaluation episodes |
| `--max-steps` | `None` | Override episode length (default: 250) |

### Smoke tests

**Random actions** -- verify the environment renders and steps correctly:

```bash
python smoke_test.py
```

**Scripted pick-and-place** -- run a Jacobian-based controller through a full pick-and-place sequence to verify the environment dynamics:

```bash
python smoke_test_pick_place.py
```

## Project structure

| File | Description |
|---|---|
| `wx250_pick_env.py` | Gymnasium environment: MJCF scene generation, physics stepping, reward, observations |
| `train_sb3_sac.py` | SAC training script with W&B logging, checkpointing, and resume support |
| `eval_policy.py` | Loads a trained model and runs episodes with the MuJoCo viewer |
| `smoke_test.py` | Minimal test: random actions in the viewer |
| `smoke_test_pick_place.py` | Scripted pick-and-place using damped least-squares IK |
| `setup_interbotix_assets.py` | Clones the public Interbotix repo to get WX250 mesh files |

## Environment details

- **Observation space (30D):** joint positions (6), joint velocities (6), gripper position (3), cube position (3), goal position (3), cube-gripper vector (3), goal-cube vector (3)
- **Action space (6D, continuous [-1, 1]):** delta targets for 5 arm joints + gripper, scaled by per-joint action scales
- **Reward:** `-0.5 * d(gripper, cube) - d(cube, goal) + 5.0 * success`
- **Success:** cube within 4 cm of goal
- **Episode length:** 250 steps (control dt = 0.02s)

## Design choices

- **Visual and collision geometry** use the Interbotix STL meshes.
- **Finger contact spheres** are added to the gripper pads for stable grasping.
- **Twin-finger gripper** is coupled with a MuJoCo equality constraint.
- **State-based observations** are used for the baseline (fastest path to a working SAC benchmark).
