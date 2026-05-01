
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.image_sac_ae import sac_ae_utils as utils


class Encoder(nn.Module):
    """Convolutional encoder used by SAC+AE-style image agents."""

    def __init__(self, obs_shape, feature_dim):
        super().__init__()
        assert len(obs_shape) == 3
        self.num_layers = 4
        self.num_filters = 32
        self.feature_dim = feature_dim

        self.convs = nn.ModuleList(
            [
                nn.Conv2d(obs_shape[0], self.num_filters, 3, stride=2),
                nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
                nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
                nn.Conv2d(self.num_filters, self.num_filters, 3, stride=1),
            ]
        )
        with torch.no_grad():
            sample = torch.zeros(1, *obs_shape)
            conv = torch.relu(self.convs[0](sample))
            for i in range(1, self.num_layers):
                conv = torch.relu(self.convs[i](conv))
            self.output_dim = conv.shape[-1]
        self.head = nn.Sequential(
            nn.Linear(self.num_filters * self.output_dim * self.output_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
        )
        self.apply(utils.weight_init)

    def forward_conv(self, obs):
        obs = obs / 255.0
        conv = torch.relu(self.convs[0](obs))
        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
        return conv

    def forward(self, obs, detach = False):
        conv = self.forward_conv(obs)
        h = conv.view(conv.size(0), -1)
        if detach:
            h = h.detach()
        out = self.head(h)
        return torch.tanh(out)


class Decoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_filters = 32, output_dim = 35):
        super().__init__()
        self.obs_shape = obs_shape
        self.num_filters = num_filters
        self.output_dim = output_dim
        self.fc = nn.Linear(feature_dim, num_filters * output_dim * output_dim)
        self.deconvs = nn.Sequential(
            nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, num_filters, 3, stride=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(num_filters, obs_shape[0], 3, stride=2, output_padding=1),
        )
        self.apply(utils.weight_init)

    def forward(self, latent):
        h = torch.relu(self.fc(latent))
        h = h.view(latent.size(0), self.num_filters, self.output_dim, self.output_dim)
        return torch.sigmoid(self.deconvs(h))


class Actor(nn.Module):
    def __init__(
        self,
        encoder,
        action_shape,
        hidden_dim,
        hidden_depth,
        log_std_bounds,
    ):
        super().__init__()
        self.encoder = encoder
        self.log_std_bounds = log_std_bounds
        self.trunk = utils.mlp(self.encoder.feature_dim, hidden_dim, 2 * action_shape[0], hidden_depth)
        self.apply(utils.weight_init)

    def forward(self, obs, detach_encoder = False):
        features = self.encoder(obs, detach=detach_encoder)
        mu, log_std = self.trunk(features).chunk(2, dim=-1)
        log_std = torch.tanh(log_std)
        log_std_min, log_std_max = self.log_std_bounds
        log_std = log_std_min + 0.5 * (log_std_max - log_std_min) * (log_std + 1)
        std = log_std.exp()
        return utils.SquashedNormal(mu, std)


class Critic(nn.Module):
    def __init__(self, encoder, action_shape, hidden_dim, hidden_depth):
        super().__init__()
        self.encoder = encoder
        self.Q1 = utils.mlp(self.encoder.feature_dim + action_shape[0], hidden_dim, 1, hidden_depth)
        self.Q2 = utils.mlp(self.encoder.feature_dim + action_shape[0], hidden_dim, 1, hidden_depth)
        self.apply(utils.weight_init)

    def forward(self, obs, action, detach_encoder = False):
        features = self.encoder(obs, detach=detach_encoder)
        h_action = torch.cat([features, action], dim=-1)
        return self.Q1(h_action), self.Q2(h_action)


@dataclass
class UpdateMetrics:
    pass


class SACAEAgent:
    """Single-image SAC+AE-style agent with a learned decoder."""

    def __init__(
        self,
        obs_shape,
        action_shape,
        action_range,
        device,
        feature_dim = 50,
        hidden_dim = 1024,
        hidden_depth = 2,
        discount = 0.99,
        init_temperature = 0.1,
        lr = 1e-3,
        decoder_lr = None,
        actor_update_frequency = 2,
        critic_tau = 0.01,
        critic_target_update_frequency = 2,
        decoder_latent_lambda = 1e-6,
        log_std_bounds = (-10, 2),
    ):
        self.device = torch.device(device)
        self.action_range = action_range
        self.discount = discount
        self.critic_tau = critic_tau
        self.actor_update_frequency = actor_update_frequency
        self.critic_target_update_frequency = critic_target_update_frequency
        self.decoder_latent_lambda = decoder_latent_lambda

        self.encoder = Encoder(obs_shape, feature_dim).to(self.device)
        self.actor = Actor(self.encoder, action_shape, hidden_dim, hidden_depth, log_std_bounds).to(self.device)
        self.critic = Critic(self.encoder, action_shape, hidden_dim, hidden_depth).to(self.device)

        target_encoder = Encoder(obs_shape, feature_dim).to(self.device)
        self.critic_target = Critic(target_encoder, action_shape, hidden_dim, hidden_depth).to(self.device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.decoder = Decoder(obs_shape, feature_dim, output_dim=self.encoder.output_dim).to(self.device)

        self.log_alpha = torch.tensor(np.log(init_temperature), device=self.device)
        self.log_alpha.requires_grad = True
        self.target_entropy = -action_shape[0]

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)
        self.decoder_optimizer = torch.optim.Adam(self.decoder.parameters(), lr=decoder_lr or lr)
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)

        self.train(True)
        self.critic_target.train()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, training = True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.decoder.train(training)

    @torch.no_grad()
    def act(self, obs, sample = False):
        obs_tensor = torch.as_tensor(obs, device=self.device).float().unsqueeze(0)
        dist = self.actor(obs_tensor, detach_encoder=False)
        action = dist.sample() if sample else dist.mean
        action = action.clamp(*self.action_range)
        return action.cpu().numpy()[0]

    def update_critic(
        self,
        obs,
        action,
        reward,
        next_obs,
        not_done,
    ):
        with torch.no_grad():
            dist = self.actor(next_obs, detach_encoder=False)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2) - self.alpha.detach() * log_prob
            target_Q = reward + not_done * self.discount * target_V

        current_Q1, current_Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        latent = self.encoder(obs, detach=False)
        reconstruction = self.decoder(latent)
        target_obs = obs / 255.0
        recon_loss = F.mse_loss(reconstruction, target_obs)
        latent_loss = 0.5 * latent.pow(2).sum(dim=1).mean()
        ae_loss = recon_loss + self.decoder_latent_lambda * latent_loss

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.decoder_optimizer.zero_grad(set_to_none=True)
        (critic_loss + ae_loss).backward()
        self.critic_optimizer.step()
        self.decoder_optimizer.step()
        return float(critic_loss.item()), float(recon_loss.item()), float(latent_loss.item())

    def update_actor_and_alpha(self, obs):
        dist = self.actor(obs, detach_encoder=True)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        actor_Q1, actor_Q2 = self.critic(obs, action, detach_encoder=True)
        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_prob - actor_Q).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_optimizer.step()

        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        self.log_alpha_optimizer.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return float(actor_loss.item()), float(alpha_loss.item()), float((-log_prob.mean()).item())

    def update(self, replay_buffer, batch_size, step):
        obs, action, reward, next_obs, not_done = replay_buffer.sample(batch_size)
        critic_loss, recon_loss, latent_loss = self.update_critic(obs, action, reward, next_obs, not_done)

        actor_loss = None
        alpha_loss = None
        entropy = None
        if step % self.actor_update_frequency == 0:
            actor_loss, alpha_loss, entropy = self.update_actor_and_alpha(obs)

        if step % self.critic_target_update_frequency == 0:
            utils.soft_update_params(self.critic, self.critic_target, self.critic_tau)

        return UpdateMetrics(
            batch_reward=float(reward.mean().item()),
            critic_loss=critic_loss,
            actor_loss=actor_loss,
            alpha_loss=alpha_loss,
            alpha_value=float(self.alpha.item()),
            entropy=entropy,
            recon_loss=recon_loss,
            latent_loss=latent_loss,
        )

    def state_dict(self):
        return {
            "encoder": self.encoder.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "critic_target": self.critic_target.state_dict(),
            "decoder": self.decoder.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "decoder_optimizer": self.decoder_optimizer.state_dict(),
            "log_alpha_optimizer": self.log_alpha_optimizer.state_dict(),
        }

    def load_state_dict(self, state):
        self.encoder.load_state_dict(state["encoder"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.critic_target.load_state_dict(state["critic_target"])
        self.decoder.load_state_dict(state["decoder"])
        self.log_alpha.data.copy_(state["log_alpha"].to(self.device))
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
        self.decoder_optimizer.load_state_dict(state["decoder_optimizer"])
        self.log_alpha_optimizer.load_state_dict(state["log_alpha_optimizer"])
