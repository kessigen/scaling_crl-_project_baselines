Image-only SAC baseline with a custom pixel encoder.

This folder uses `image_sac/` untouched and keeps the same image-only
observation pipeline, but replaces SB3's default `NatureCNN` with a more
pixel encoder:

- 4 convolutional layers with 32 channels
- compact latent projection with `LayerNorm` and `tanh`
-  SB3 SAC actor/critic heads on top

This is still SAC in SB3 but different encoder
