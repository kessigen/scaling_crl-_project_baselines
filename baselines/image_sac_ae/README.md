Paper-style SAC+AE baseline adapted to this project.

This folder keeps the same WX250 image-only task as `image_sac/`, but uses a
custom PyTorch implementation instead of SB3's stock pixel SAC path:

- single-image observations (no frame stack by default)
- custom convolutional encoder for actor/critic
- decoder trained with a reconstruction loss
- learned entropy temperature
- actor updates with detached encoder features

This is intended to be a separate SAC+AE-style baseline for the repo, not a
modification of `image_sac/`.
