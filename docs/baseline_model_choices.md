# Baseline Model Changes and Design Choices

This document summarizes the baseline implementations added for the WidowX/WX250 pick-and-place  over the course of the project. The prosgreesion over multiplevariantsled to best implementaion for our experiments

## Implementation Order

The implemented baselines follow this progression:

1. Pose-based SAC
2. Image-based SAC
3. Image-based SAC with a custom pixel encoder
4. Image-based SAC with an autoencoder objective
5. DrQ image-based sac
6. In-zone-success variants for final-position evaluation

.

## Pose-Based SAC

Folder:

```text
baselines/pose_sac_old/
```

Pose SAC is the simplest baseline. It uses Stable-Baselines3 SAC with an `MlpPolicy` on low-dimensional pose/state observations from:

```text
envs/wx250_pick_env.py
```

Initial  MuJoCo task, action space, reward, reset logic, and checkpointing before adding image complexity. more complex than actual task. reuses mujoco menagerie code
Key choices:

- Uses state/pose observations instead of pixels.
- Uses SB3 SAC framework
- Saves checkpoints, best model, final model, replay buffer, and W&B metrics under `runs/wx250_pose_sac`.
- note: Uses the  checkpoint prefix `wx250_pose_sac`.

## Image-Based SAC

Folder:

```text
baselines/image_sac/
```

Image SAC extends the same task to visual observations from:

```text
envs/wx250_pick_env_image.py
```

This baseline uses Stable-Baselines3 SAC with `CnnPolicy` and frame stacking. It is the first image-only baseline and acts as the stock pixel-SAC comparison point.

Key choices:

- Uses RGB image observations instead of low-dimensional pose features.
- Uses SB3's default CNN policy path for a simple image baseline.
- Applies vectorized environment wrapping and frame stacking through SB3 utilities.(tried up to 8 simultaneous envs. negligible speed up + too expensive on A100)
- Supports camera selection through `--obs-camera`, with `front`, `isometric`, and `topdown` camera options from the environment.
- Saves outputs under `runs/wx250_image_sac`.
- Uses the  checkpoint prefix `wx250_image_sac`.

## Image-Based SAC With Custom Pixel Encoder

Folder:

```text
baselines/image_sac_pixel_encoder/
```

The custom encoder baseline keeps the SB3 SAC training loop but replaces SB3's default image feature extractor with a smaller pixel-RL-style encoder:

```text
baselines/image_sac_pixel_encoder/custom_pixel_encoder.py
```

 vanilla SAC variant with custom encoder, but the representation module is closer to encoders used in pixel-based RL papers.(SAC autoencoder encoder logic without reconstruction)

Key choices:

- Keeps SB3 SAC for comparability with `image_sac`.
- Replaces the default CNN with a custom `PixelEncoder`.
- Uses four convolutional layers with 32 channels.
- Projects to a compact latent feature vector.
- Applies `LayerNorm` and `tanh` after the latent projection.
- Saves outputs under `runs/wx250_image_sac_pixel_encoder`.
- Uses the  checkpoint prefix `wx250_image_sac_pixel_encoder`.

## Image-Based SAC With Autoencoder

Folder:

```text
baselines/image_sac_ae/
```

The SAC+AE baseline moves away from the SB3  training path and uses a custom PyTorch implementation. It adds additional reconstruction objective.Hoped to get good latent for compression . too expensive


Key choices:

- Uses a custom PyTorch SAC+AE-style agent instead of SB3 SAC.
- Uses image observations from `WX250PickPlaceImageEnv`.
- Converts images into channel-first format with `ImageToCHW`.
- Uses a shared convolutional encoder for actor and critic features.
- Adds a decoder and reconstruction loss.
- Uses a learned entropy temperature.
- Detaches encoder features during actor updates, matching the usual SAC+AE design choice.
- Saves outputs under `runs/wx250_image_sac_ae`.

## DrQ

Folder:

```text
baselines/image_drq/
```

The DrQ baseline is the  image-based baseline with best result overall for all image models. It uses a  PyTorch implementation  of sac(noSB3). DrQ relies on image augmentation, especially random shifts, instead of reconstruction loss.

This baseline was implemented last because it combines a custom encoder, custom actor/critic updates, replay buffer handling, augmentation, target updates, a

Key choices:

- Uses a custom PyTorch DrQ-style agent.
- Uses image observations from `WX250PickPlaceImageEnv`.
- Uses `DrQFrameStack` for stacked image observations.
- Uses a custom replay buffer.
- Applies random-shift augmentation.
- Uses two augmented target views for critic targets.
- Updates the actor every 2 steps.
- Updates the target critic every 2 steps.
- Uses a learned entropy temperature.
- Supports resume from `.pt` checkpoints.
- Saves outputs under `runs/wx250_image_drq`.

## In-Zone-Success Variants - Selected Baselines

These are the two baselines reported in the project paper as **SAC (MLP)** with
pose representation and **SAC + DRQ** with image representation (Tables 1 and 2).

Folders:

```text
baselines/sac_pose/    # report: SAC (MLP) / pose
baselines/sac_drq/     # report: DRQ(SAC + Augmentations) / image
```

The in-zone-success variants use:

```text
envs/wx250_pick_env_in_zone.py
```

This environment changes the success semantics. Instead of terminating immediately when the cube reaches the goal, the episode continues until the horizon. The final success signal therefore measures whether the cube is in-zone at the end of the episode.

This was added because immediate-success termination can overstate performance when a policy briefly touches the goal but does not keep the object there.

Key choices:

- Uses non-terminating success semantics.
- Tracks `is_success`, `consecutive_in_zone`, `total_in_zone_steps`, and `in_zone_fraction`.
- Keeps pose and image variants separate.
- Provides a pose SAC counterpart for the in-zone metric.
- Provides a DrQ image counterpart for the in-zone metric.
- Saves in-zone pose SAC outputs under `runs/wx250s_in_zone_pose_sac`.
- Saves in-zone DrQ outputs under `runs/wx250s_in_zone_image_drq`.

## Shared Environment and Training Choices

Across the baselines, the implementation keeps task definitions and naming consistent:

- Pose-based baselines use `WX250PickPlaceEnv` or the in-zone environment in `obs_mode="pose"`.
- Image-based baselines use `WX250PickPlaceImageEnv` or `WX250PickPlaceImageInZoneEnv`.
- Image baselines support the same camera names: `front`, `isometric`, and `topdown`.
- Checkpoints and run directories now use consistent names that match their folder names.
- Baseline folders are independent entry points, so each train/eval script can be run from its own folder path.
- Shared code is imported by explicit package names such as `baselines.image_drq` and `baselines.image_sac_ae`.

## What Each Baseline Tests

| Baseline | Observation | Main Question |
| --- | --- | --- |
| `pose_sac` | Pose/state | Can SAC solve the task when perception is not the bottleneck? |
| `image_sac` | RGB images | How well does stock SB3 SAC learn directly from pixels? |
| `image_sac_pixel_encoder` | RGB images | Does a custom pixel-RL encoder improve over the default CNN? |
| `image_sac_ae` | RGB images | Does reconstruction-based representation learning help control? |
| `image_drq` | RGB images | Does augmentation-based pixel RL improve robustness and especially sample efficiency? |
| `sac_pose` (paper: SAC (MLP)) | Pose/state | Selected baseline: can pose SAC keep the cube in-zone at the center at the end of the episode? |
| `sac_drq` (paper: SAC + DRQ) | RGB images | Selected baseline: can DrQ learn the stricter in-zone-at-end success objective from pixels? |

## Summary

The baseline suite was structured to make comparisons fair and interpretable. The models are not just separate folders; they represent a deliberate ladder of difficulty:

```text
Timeline:
pose SAC -> image SAC -> image SAC + custom encoder -> SAC+AE -> DrQ
```

This makes it easier to explain whether performance changes come from the observation type, the encoder design, the auxiliary reconstruction objective, or DrQ-style augmentation.
