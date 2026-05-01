# CLI flag reference

These are all flags i added during my various implementations for faster  comparison of multiple runs
Full flag tables for the two selected baselines (`sac_pose`, `sac_drq`).
Variants share the same conventions - run `python -m baselines.<name>.train --help`
for their exact lists.

---

## Shared env knobs (all four selected-baseline scripts)

These configure `WX250PickPlaceImageInZoneEnv`. They **must match between train
and eval** for results to be apples-to-apples.

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--red-zone-x` | float | `0.30` | Red (spawn) zone center, X |
| `--red-zone-y` | float | `-0.12` | Red zone center, Y |
| `--blue-zone-x` | float | `0.30` | Blue (goal) zone center, X |
| `--blue-zone-y` | float | `0.12` | Blue zone center, Y |
| `--zone-half` | float | `0.06` | Visible zone marker half-extent (m). Must be >= `success-threshold`. |
| `--spawn-noise` | float | `0.02` | Cube spawn jitter inside the red zone |
| `--goal-noise` | float | `0.02` | Goal jitter inside the blue zone |
| `--success-threshold` | float | `0.04` | XY half-extent of the in-zone-success region (m) |
| `--park-bonus` | float | `2.0` | Per-step reward while cube is inside the success region |
| `--max-steps` | int | `200` | Episode horizon (steps) |
| `--randomize-eval` | flag | off | Domain-randomize the eval env (lighting / cam jitter) |

Train scripts also accept `--randomize-train` (same shape, applies to the train env).

---

## `baselines.sac_pose.train` - paper "SAC (MLP)" / pose

```bash
python -m baselines.sac_pose.train [--flag ...]
```

### Training loop

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--timesteps` | int | `300_000` | Total env steps |
| `--run-dir` | str | `runs/wx250s_in_zone_pose_sac` | Output dir for ckpts, eval logs, W&B |
| `--seed` | int | `1` | RNG seed |
| `--device` | str | `cuda` if available else `cpu` | Torch device |

### SAC hyperparameters

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--buffer-size` | int | `250_000` | Replay buffer capacity |
| `--batch-size` | int | `256` | Minibatch size |
| `--lr` | float | `3e-4` | Learning rate (actor / critic / Î±) |
| `--tau` | float | `0.005` | Target-net soft-update rate |
| `--gamma` | float | `0.99` | Discount |
| `--learning-starts` | int | `10_000` | Env steps before SAC updates begin |
| `--init-temperature` | float | `0.1` | Initial SAC entropy coefficient Î± |
| `--alpha-floor` | float | `0.05` | Minimum Î± (clamp via `AlphaFloorCallback` so exploration can't collapse) |

### Eval / checkpointing / logging

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--save-freq` | int | `50_000` | Checkpoint period (steps) |
| `--eval-freq` | int | `10_000` | Eval-callback period (steps) |
| `--eval-episodes` | int | `10` | Episodes per eval pass |
| `--log-freq` | int | `1000` | W&B / console log period |
| `--no-wandb` | flag | off | Disable W&B |
| `--wandb-dir` | str | `/tmp/wandb` | W&B local dir |
| `--resume` | flag | off | Auto-resume from latest `wx250_pose_sac_*_steps.zip` (or `final_model.zip`) in `--run-dir` |

---

## `baselines.sac_drq.train` - paper "SAC + DRQ" / image

```bash
python -m baselines.sac_drq.train [--flag ...]
```

### Training loop

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--timesteps` | int | `300_000` | Total env steps |
| `--run-dir` | str | `runs/wx250s_in_zone_image_drq` | Output dir |
| `--seed` | int | `1` | RNG seed |
| `--device` | str | `cuda` if available else `cpu` | Torch device |

### Observation

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--obs-size` | int | `64` | Square obs side (px); frame-stack k=3 |
| `--obs-camera` | choice | `front` | `front` / `isometric` / `topdown` |

### DrQ / SAC hyperparameters

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--buffer-size` | int | `500_000` | Replay buffer capacity |
| `--saved-replay-buffer-size` | int | `8_000` | Transitions to dump to disk per checkpoint |
| `--batch-size` | int | `256` | Minibatch size |
| `--image-pad` | int | `4` | Random-shift pad (px) for DrQ augmentation |
| `--feature-dim` | int | `50` | Encoder output dim |
| `--hidden-dim` | int | `1024` | Actor / critic MLP hidden width |
| `--hidden-depth` | int | `2` | MLP hidden depth |
| `--lr` | float | `3e-4` | Learning rate |
| `--discount` | float | `0.99` | Discount Î³ |
| `--critic-tau` | float | `0.01` | Target-critic soft-update rate |
| `--init-temperature` | float | `0.1` | Initial Î± |
| `--actor-update-frequency` | int | `2` | Actor update every N steps |
| `--critic-target-update-frequency` | int | `2` | Target-critic update every N steps |
| `--seed-steps` | int | `1000` | Random-action warmup before training |

### Eval / checkpointing / logging

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--eval-freq` | int | `10_000` | Eval period (steps) |
| `--eval-episodes` | int | `10` | Episodes per eval pass |
| `--save-freq` | int | `50_000` | Best-model save period |
| `--log-freq` | int | `1000` | W&B / console log period |
| `--checkpoint-freq` | int | `None` (= `--save-freq`) | Override checkpoint period |
| `--no-checkpoints` | flag | off | Skip periodic checkpoints |
| `--no-replay-buffer` | flag | off | Don't dump replay buffer alongside checkpoints (saves disk) |
| `--no-wandb` | flag | off | Disable W&B |
| `--wandb-dir` | str | `/tmp/wandb` | W&B local dir |
| `--resume` | str | `None` | **Path** to a `.pt` checkpoint to resume from (differs from `sac_pose.train` - that one is a boolean auto-find) |

---

## `baselines.sac_pose.eval`

```bash
python -m baselines.sac_pose.eval --model PATH/TO/best_model.zip [--flag ...]
```

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--model` | str | **required** | SB3 SAC `.zip` checkpoint |
| `--n-episodes` | int | `20` | Episodes to roll out |
| `--device` | str | `auto` | Torch device |

Plus the shared env knobs above and the GIF flags below.

---

## `baselines.sac_drq.eval`

```bash
python -m baselines.sac_drq.eval --model PATH/TO/best_model.pt [--flag ...]
```

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--model` | str | **required** | DrQ `.pt` checkpoint |
| `--n-episodes` | int | `20` | Episodes to roll out |
| `--device` | str | `cuda` if available else `cpu` | Torch device |
| `--obs-size` | int | `64` | Must match training |
| `--obs-camera` | choice | `front` | Must match training |
| `--feature-dim` | int | `50` | Must match training |
| `--hidden-dim` | int | `1024` | Must match training |
| `--hidden-depth` | int | `2` | Must match training |

Plus the shared env knobs and the GIF flags below.

---

## GIF recording (both eval scripts)

| Flag | Type | Default | Meaning |
| --- | --- | --- | --- |
| `--record-gif PATH` | str | `None` | Save one episode chosen by `--record-gif-mode` |
| `--record-gif-mode` | choice | `random` | `random` / `best` / `median` / `random-success` / `best-success` / `median-success` |
| `--record-best-gif PATH` | str | `None` | Always saves the best-return episode |
| `--record-success-gif PATH` | str | `None` | Save a success-only episode |
| `--record-success-gif-mode` | choice | `random-success` | `random-success` / `best-success` / `median-success` |
