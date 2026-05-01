# WidowX-250 Pick-and-Place: SAC Baselines

This repository contains the **SAC-family baselines** used in the project
report *"Bridging Simulation and Reality with Ultra-Deep Self-Supervised
Reinforcement Learning"* (S. Rajkumar and K. Ramasubboo, IFT6163, April 2026).

It provides:

- MuJoCo simulation environments for the WidowX-250 cube pick-and-place task
- The two SAC baselines reported in the paper, plus the variants we tried
  before settling on them
- Smoke tests and camera sanity checks
- The Interbotix mesh assets fetcher

The contrastive RL (CRL / deep CRL) implementation lives in a separate
repository 

---

## Selected baselines (report Tables 1 and 2)

The two baselines reported in the paper are both trained on the
**in-zone-success** environment (`envs/wx250_pick_env_in_zone.py`), where the
episode runs to the horizon and success is measured by whether the cube is
in-zone at the end of the episode (rather than terminating
on first contact with the goal).

| Folder | Paper name | Representation | Algorithm |
| --- | --- | --- | --- |
| `baselines/sac_pose/` | **SAC (MLP)** | Pose / state | SB3 SAC with `MlpPolicy` |
| `baselines/sac_drq/` | **SAC + DRQ** | RGB images | Custom PyTorch DrQ-v2-style SAC with random-shift augmentation |

Reported sim performance (cube pushing, 20 episodes, Table 1):

| Method | Representation | Final SR |
| --- | --- | --- |
| SAC (MLP) | Pose | 90.0% |
| SAC + DRQ | Image | 50.0% |

### Train

```bash
# SAC (MLP) - pose representation
python -m baselines.sac_pose.train --timesteps 300000 --no-wandb

# SAC + DRQ - image representation
python -m baselines.sac_drq.train --timesteps 300000 --no-wandb
```

Each script writes checkpoints, eval logs, and (optionally) W&B runs to its
own `runs/` subdirectory:

- `runs/wx250s_in_zone_pose_sac/` for `sac_pose`
- `runs/wx250s_in_zone_image_drq/` for `sac_drq`

Use `--help` on either entry point for the full list of env knobs (zone
positions, success threshold, park-bonus, max steps) and training knobs
(buffer size, learning rate, eval frequency, etc.). For a complete flag
reference, see [`docs/cli_flags.md`](docs/cli_flags.md).

### Evaluate / record GIFs

```bash
python -m baselines.sac_pose.eval --model runs/wx250s_in_zone_pose_sac/best_model/best_model.zip
python -m baselines.sac_drq.eval  --model runs/wx250s_in_zone_image_drq/best_model.pt

# add --record-gif out.gif (or --record-success-gif) to save a rollout
```

### Loading checkpoints from before the rename

The folder rename (`baselines/in_zone_success/{pose_sac,image_drq}` ->
`baselines/{sac_pose,sac_drq}`) does **not** affect the saved checkpoint
files:

- `sac_pose` writes SB3 SAC `.zip` files; `SAC.load(path)` is path-agnostic.
- `sac_drq` writes `torch.save({"agent": agent.state_dict(), ...})`; the
  `DRQAgent` class itself still lives at `baselines.image_drq.drq_agent`
  and is reused by the new entry point.

Replay-buffer pickle files saved alongside checkpoints continue to load for
the same reason - the `ReplayBuffer` class path is unchanged.

---

## Variants we tried (not the selected baselines)

The folders below are earlier image-encoder and SAC-variant experiments. They
informed the choice to use DrQ for the image baseline (see Section 4.2.3 of
the paper), but only `sac_pose` and `sac_drq` are reported. Each folder is a
self-contained entry point.

| Folder | Description |
| --- | --- |
| `baselines/pose_sac_old/` | Earlier SB3 SAC + `MlpPolicy` on the non-in-zone env (`wx250_pick_env.py`). |
| `baselines/image_sac/` | SB3 SAC with the default `CnnPolicy` on `wx250_pick_env_image.py`. |
| `baselines/image_sac_pixel_encoder/` | SB3 SAC with a custom pixel-RL encoder replacing the default `NatureCNN`. |
| `baselines/image_sac_ae/` | Custom PyTorch SAC + autoencoder (reconstruction-loss representation learning). |
| `baselines/image_drq/` | Custom PyTorch DrQ on the non-in-zone env. **Also serves as the implementation library** that `sac_drq/` imports (`drq_agent.py`, `drq_replay_buffer.py`, `drq_utils.py`, `frame_stack.py`). |

For the rationale behind each variant and the order they were tried, see
[`docs/baseline_model_choices.md`](docs/baseline_model_choices.md).

### Variant train and eval commands

Each variant is a self-contained entry point with its own `--run-dir` default;
the eval scripts take `--model PATH`. Run any of them with `--help` for the
full flag list.

```bash
# Pose SAC (non-in-zone env)
python -m baselines.pose_sac_old.train --timesteps 300000 --no-wandb
python -m baselines.pose_sac_old.eval  --model runs/wx250_pose_sac/best_model/best_model.zip

# Image SAC (SB3 default CnnPolicy)
python -m baselines.image_sac.train --timesteps 1500000 --no-wandb
python -m baselines.image_sac.eval  --model runs/wx250_image_sac/best_model/best_model.zip

# Image SAC + custom pixel encoder
python -m baselines.image_sac_pixel_encoder.train --timesteps 1500000 --no-wandb
python -m baselines.image_sac_pixel_encoder.eval  --model runs/wx250_image_sac_pixel_encoder/best_model/best_model.zip

# Image SAC + autoencoder
python -m baselines.image_sac_ae.train --timesteps 300000 --no-wandb
python -m baselines.image_sac_ae.eval  --model runs/wx250_image_sac_ae/best_model.pt

# DrQ on the non-in-zone env (this folder also provides the agent code reused by sac_drq)
python -m baselines.image_drq.train --timesteps 300000 --no-wandb
python -m baselines.image_drq.eval  --model runs/wx250_image_drq/best_model.pt
```

---

## Repository structure

```text
envs/
  wx250_pick_env.py          # pose/state observation environment
  wx250_pick_env_image.py    # image or pose observation environment
  wx250_pick_env_in_zone.py   # in-zone-success variant (used by selected baselines)

baselines/
  sac_pose/                  # *** selected baseline: paper "SAC (MLP)" / pose
  sac_drq/                   # *** selected baseline: paper "SAC + DRQ" / image
  pose_sac_old/              # variant: SB3 pose SAC on non-in-zone env
  image_sac/                 # variant: SB3 default CnnPolicy
  image_sac_pixel_encoder/   # variant: SB3 SAC with custom pixel encoder
  image_sac_ae/              # variant: custom SAC + autoencoder
  image_drq/                 # variant + DRQ implementation library reused by sac_drq

scripts/
  smoke_test_pick_place.py   # pose env smoke test
  smoke_test_pose_env.py     # in-zone pose env smoke test
  smoke_test_image.py        # image env smoke test
  smoke_test_image_pixel.py  # pixel-encoder smoke test
  smoke_test_drq.py          # DrQ smoke test
  smoke_test_sac_ae.py       # SAC+AE smoke test
  sanity_cam.py              # camera-pose sanity render

docs/
  baseline_model_choices.md  # rationale and design notes for every baseline

assets/external/             # Interbotix mesh assets (fetched on demand)
results/                     # metrics CSVs and saved GIF outputs
```

---

## Setup

```bash
pip install -r requirements.txt
```

The Interbotix assets are expected under `assets/external/interbotix_ros_manipulators`.
If they are missing, run:

```bash
python setup_interbotix_assets.py
```

On Linux, the train/eval scripts default to `MUJOCO_GL=egl` for headless
rendering. On Windows / macOS the default OpenGL backend is used.

---

## Quick smoke checks

```bash
# Render a few cameras and verify the env loads
python scripts/sanity_cam.py

# Step through the in-zone env with random actions
python scripts/smoke_test_pose_env.py

# Run a few DrQ updates on tiny obs to verify the agent compiles end-to-end
python scripts/smoke_test_drq.py
```

---

## Reward / success definition

Both selected baselines share the same shaped reward and success criterion
(paper Section 4.2.1). Briefly:

- `r_reach = 1 - tanh(8 * ||x_grip - x_cube||)` - encourages reaching the cube.
- `r_push  = 1 - tanh(6 * ||x_cube,xy - x_goal,xy||)` - encourages pushing the
  cube toward the goal in the XY plane, gated by a contact term so it only
  activates when the gripper is near the cube.
- A constant `-0.04` per-step decay prevents hovering local minima.
- A small `park_bonus` is added for every step the cube is inside the
  Chebyshev-distance success region around the goal.
- An episode is an **in-zone-at-end success** if the cube is inside the success
  region (default `success_threshold = 0.04 m`) at the final step.

The eval scripts report `success_rate` (in-zone at end), mean return, mean
episode length, and `in_zone_fraction` (fraction of steps spent inside the
success region).

---

## Project paper and companion repos

- Paper: *Bridging Simulation and Reality with Ultra-Deep Self-Supervised
  Reinforcement Learning*, Rajkumar & Ramasubboo, IFT6163, April 2026.
- Contrastive RL (CRL / deep CRL) implementation:
  <https://github.com/suryaprakashrajkumar/CRL>
- This repo (SAC baselines):
  <https://github.com/kessigen/scaling_crl-_project_baselines>
- Sim eval videos:
  <https://drive.google.com/drive/folders/1LlmID9xoCP7BSpit_VmYGolqVc-tC3_B?usp=sharing>

scratch notes from  building this are in [LOG.md](LOG.md) -
roughly chronological, edited to make them mre readable
