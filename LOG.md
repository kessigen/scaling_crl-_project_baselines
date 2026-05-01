# project log

scratch notes from building the SAC baselines. mostly chronological,
not edited for style. main repo for paper-side code is the CRL one.

## first pose env runs

basic WX250PickPlaceEnv up. SB3 SAC + MlpPolicy, 200k steps, no in-zone
semantics yet. reward was just `-||cube - goal||` linear, agent learned
to reach the cube after ~60k but then just hovered next to it. no actual
pushing.

added a contact-gated push term (`r_push * exp(-(20*dgrip)^2)`). same
problem - hover, but now closer to the cube.

## first park-fraction metric

added `in_zone_fraction` to info dict so wandb shows whether the cube
*stays* in the goal vs touches it once and bounces. 

## reward shaping converged

tried a few variants:

- linear ||.||  : too flat for SAC, no signal
- tanh(8*dgrip) : works for reaching but over-rewards hovering
- + per-step decay (-0.04) : finally breaks the hover local min

final form is in section 4.2.1 of the paper. the -0.04 was the real
unlock - everything before it kept hovering.

## pose SAC works

90% SR on cube push with pose obs at 300k steps, 3-layer 256-unit MLP.
this is roughly what made it into the paper as SAC (MLP).

## image SAC, first attempt

SB3 SAC + CnnPolicy + frame stack k=4. 1.5M steps. SR plateaued
at ~30%. extremely sample inefficient. NatureCNN is too generic for the
small 64x64 isometric obs we're using.

## image SAC + custom pixel encoder

swapped the feature extractor for a 4x32 conv stack with LayerNorm + tanh
on the latent. marginal improvement (~35% SR at 1.5M). not worth the
custom code.

## SAC + AE

custom torch agent, decoder + reconstruction loss, detached encoder
features for the actor. recon loss spiked when obs_size=84 (probably
gradient flowing back into the conv stack from both objectives at once),
dropped to 64 and it stabilised. final SR ~40-45% at 300k steps. better
than image SAC but still not great.

used ~30GB of disk per ckpt because of the replay buffer dump. added
`--saved-replay-buffer-size` 
## DrQ first run

random-shift augmentation + 2 augmented target views. 50% SR at 200k
steps - already comparable to SAC+AE but cheaper to train. this is what
ended up being the image baseline in the paper.

NOTE:

- alpha collapsed to ~1e-4 around step 80k, exploration died. fix:
  floor the log_ent_coef at log(0.05). this is the AlphaFloor
  callback now used in both DrQ and the pose baseline.
- critic loss spiked when image_pad was 8. dropped to 4 . 
- feature_dim=128 better than 512 and 1024 variants.256 is better after more steps  

## in-zone-success env

immediate-success termination was hurting traning. policy that
touches the goal and bounces gets 100%  but might
have in_zone_fraction = 0.05. switched to in-zone bonus - episode
runs to horizon and success is "is the cube in the zone at the final
step". no loger allows throwing to zone

re-ran SAC (MLP) and SAC + DrQ on the in-zone env. these are the
selected baselines now in `baselines/sac_pose/` and `baselines/sac_drq/`.

## sim2real first attempt

zero-shot transfer to the real WidowX-250 with isometric Realsense
D435 cam at the same pose as sim. did not work well. lighting is the
biggest gap visually but even the pose policy fails because we don't
have ground-truth cube xy on the real arm - we'd need a marker or
detector .

