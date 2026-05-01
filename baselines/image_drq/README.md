Paper-style DrQ baseline adapted to this project.

This folder keeps the same task and image-only observation setting as
`image_sac/`, but replaces the SB3 SAC path with a custom PyTorch DrQ agent
that follows the original paper structure more closely:

- custom encoder, actor, and critic
- learned entropy temperature
- actor update every 2 steps
- target critic update every 2 steps
- random-shift augmentation with two augmented target views

The task and reward are still your WX250 pick-and-place environment, so this is
not a benchmark reproduction of the original paper results. It is a paper-like
DrQ implementation for your environment.
