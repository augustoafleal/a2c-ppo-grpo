import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.distributions import Categorical


class A2CBase(nn.Module):
    def __init__(
        self,
        n_features,
        n_actions,
        device,
        critic_lr,
        actor_lr,
        n_envs,
        atari_mode,
        ppo_epochs=None,
        ppo_batch_size=None,
        clip_coef=None,
        stack_size=4,
        use_kl=None,
        is_continuous_action=False,
    ):

        super().__init__()
        self.device = device
        self.n_envs = n_envs
        self.atari_mode = atari_mode
        self.stack_size = stack_size
        self.n_actions = n_actions
        self.actor_lr = actor_lr
        self.n_features = n_features
        self.use_kl = use_kl
        self.continuous_action = is_continuous_action
        if atari_mode:
            self.conv1 = nn.Conv2d(in_channels=stack_size, out_channels=16, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)

            self.flattened_size = 32 * 9 * 9
            self.flatten = nn.Flatten()

            self.fc = nn.Sequential(nn.Linear(self.flattened_size, 256), nn.ReLU()).to(device)

            self.critic = nn.Linear(256, 1).to(device)
            self.actor = nn.Linear(256, n_actions).to(device)
            self.feature_extractor_params = list(self.conv1.parameters()) + list(self.conv2.parameters())

            self.critic_optim = optim.RMSprop(
                list(self.critic.parameters()) + self.feature_extractor_params, lr=critic_lr
            )
            self.actor_optim = optim.RMSprop(list(self.actor.parameters()) + self.feature_extractor_params, lr=actor_lr)

            self.optim = optim.RMSprop(
                [
                    {"params": self.conv1.parameters()},
                    {"params": self.conv2.parameters()},
                    {"params": self.fc.parameters()},
                    {"params": self.actor.parameters()},
                    {"params": self.critic.parameters()},
                ],
                lr=actor_lr,
                alpha=0.99,
                eps=1e-5,
            )

        else:

            self.hidden_size = (64, 64)

            self.critic = nn.Sequential(
                nn.Linear(n_features, self.hidden_size[1]),
                nn.Tanh(),
                nn.Linear(self.hidden_size[0], self.hidden_size[1]),
                nn.Tanh(),
                nn.Linear(self.hidden_size[0], 1),
            ).to(device)

            self.actor = nn.Sequential(
                nn.Linear(n_features, self.hidden_size[1]),
                nn.Tanh(),
                nn.Linear(self.hidden_size[0], self.hidden_size[1]),
                nn.Tanh(),
                nn.Linear(self.hidden_size[0], n_actions),
            ).to(device)

            self.critic_optim = optim.Adam(self.critic.parameters(), lr=critic_lr)
            self.actor_optim = optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.optim = optim.Adam(
                [
                    {"params": self.actor.parameters(), "lr": actor_lr},
                    {"params": self.critic.parameters(), "lr": critic_lr},
                ],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0,
            )
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.clip_coef = clip_coef

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)

        if self.atari_mode:
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.flatten(x)
            x = self.fc(x)
            critic_out = self.critic(x)
            actor_out = torch.softmax(self.actor(x), dim=-1)
        else:
            critic_out = self.critic(x)
            actor_out = self.actor(x)

        return critic_out, actor_out

    def select_action(self, x):
        value, logits = self.forward(x)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return actions, log_probs, value, entropy

    def update_parameters(self, critic_loss, actor_loss):
        self.optim.zero_grad()
        total_loss = critic_loss + actor_loss
        total_loss.backward()
        self.optim.step()


class A2CSimple(A2CBase):
    def __init__(self, *args, is_continuous_actions=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.is_continuous_actions = is_continuous_actions

        if self.is_continuous_actions:
            self.actor = nn.Sequential(
                nn.Linear(self.n_features, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, self.n_actions),
                nn.Tanh(),
            ).to(self.device)
            self.log_std = nn.Parameter(torch.zeros(self.n_actions, device=self.device))
            self.critic = nn.Sequential(
                nn.Linear(self.n_features, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, 1),
            ).to(self.device)
            self.optim = optim.Adam(
                [
                    {"params": self.actor.parameters()},
                    {"params": self.critic.parameters()},
                    {"params": [self.log_std]},
                ],
                lr=self.actor_lr,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

    def select_action(self, x):
        value, logits = self.forward(x)

        if self.is_continuous_actions:
            mean = logits
            std = self.log_std.exp()
            dist = torch.distributions.Normal(mean, std)

            raw_action = dist.rsample()
            action = torch.tanh(raw_action)
            gaussian_logp = dist.log_prob(raw_action).sum(-1)
            squash_correction = torch.log(1 - action.pow(2) + 1e-6).sum(-1)
            log_prob = gaussian_logp - squash_correction
            entropy = dist.entropy().sum(-1)

            return action, log_prob, value, entropy, raw_action

        else:
            dist = torch.distributions.Categorical(logits=logits)

            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            return action, log_prob, value, entropy, logits

    def update_agent(self, rollouts, hp):
        if self.atari_mode:
            T, N, C, H, W = rollouts["states"].shape
            obs_dim = C * H * W
        else:
            T, N, obs_dim = rollouts["states"].shape

        device = rollouts["states"].device

        advantages = torch.zeros(T, N, device=device)
        gae = 0.0

        for t in reversed(range(T - 1)):
            td_error = (
                rollouts["rewards"][t]
                + hp["gamma"] * rollouts["value_preds"][t + 1] * rollouts["masks"][t]
                - rollouts["value_preds"][t]
            )
            gae = td_error + hp["gamma"] * hp["lam"] * rollouts["masks"][t] * gae
            advantages[t] = gae

        returns = advantages + rollouts["value_preds"]

        advantages_flat = advantages.reshape(-1).detach()
        advantages_flat = (advantages_flat - advantages_flat.mean()) / (advantages_flat.std() + 1e-8)

        returns_flat = returns.reshape(-1).detach()

        if self.is_continuous_actions:
            actions_flat = rollouts["pre_tanh_actions"].reshape(-1, self.n_actions)
        else:
            actions_flat = rollouts["actions"].reshape(-1)

        if self.atari_mode:
            states_flat = rollouts["states"].reshape(-1, hp["stack_size"], 84, 84)
        else:
            states_flat = rollouts["states"].reshape(-1, obs_dim)

        values, new_logits = self.forward(states_flat)
        values = values.squeeze(-1)

        critic_loss = (returns_flat.detach() - values).pow(2).mean()

        if self.is_continuous_actions:
            mean = new_logits
            std = self.log_std.exp()
            dist = torch.distributions.Normal(mean, std)

            gaussian_logp = dist.log_prob(actions_flat).sum(-1)

            action_tanh = torch.tanh(actions_flat)
            squash_correction = torch.log(1 - action_tanh.pow(2) + 1e-6).sum(-1)

            log_probs = gaussian_logp - squash_correction

            entropy = dist.entropy().sum(-1).mean()

        else:
            dist = torch.distributions.Categorical(logits=new_logits)
            log_probs = dist.log_prob(actions_flat)
            entropy = dist.entropy().mean()

        actor_loss = -(advantages_flat.detach() * log_probs).mean() - hp["ent_coef"] * entropy

        self.update_parameters(critic_loss, actor_loss)

        return critic_loss.item(), actor_loss.item(), entropy.item()


class PPO(A2CBase):
    def __init__(
        self, *args, ppo_epochs=None, ppo_batch_size=None, clip_coef=None, is_continuous_actions=False, **kwargs
    ):
        super().__init__(*args, **kwargs)

        if ppo_epochs is None:
            raise ValueError("ppo_epochs must be provided for PPO")
        if ppo_batch_size is None:
            raise ValueError("ppo_batch_size must be provided for PPO")
        if clip_coef is None:
            raise ValueError("clip_coef must be provided for PPO")

        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.clip_coef = clip_coef

        self.is_continuous_actions = is_continuous_actions

        if self.atari_mode:
            self.conv1 = nn.Conv2d(in_channels=self.stack_size, out_channels=16, kernel_size=8, stride=4).to(
                self.device
            )
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2).to(self.device)
            self.flattened_size = 32 * 9 * 9
            self.flatten = nn.Flatten()
            self.fc = nn.Sequential(nn.Linear(self.flattened_size, 256), nn.ReLU()).to(self.device)

            self.actor = nn.Linear(256, self.n_actions).to(self.device)
            self.critic = nn.Linear(256, 1).to(self.device)

            self.feature_extractor_params = list(self.conv1.parameters()) + list(self.conv2.parameters())

            self.optim = optim.RMSprop(
                [
                    {"params": self.conv1.parameters()},
                    {"params": self.conv2.parameters()},
                    {"params": self.fc.parameters()},
                    {"params": self.actor.parameters()},
                    {"params": self.critic.parameters()},
                ],
                lr=getattr(self, "actor_lr", 7e-4),
                alpha=0.99,
                eps=1e-5,
            )
        else:
            if self.is_continuous_actions:
                self.actor = nn.Sequential(
                    nn.Linear(self.n_features, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, self.n_actions),
                    nn.Tanh(),
                ).to(self.device)
                self.log_std = nn.Parameter(torch.zeros(self.n_actions, device=self.device))
                self.critic = nn.Sequential(
                    nn.Linear(self.n_features, 64),
                    nn.Tanh(),
                    nn.Linear(64, 64),
                    nn.Tanh(),
                    nn.Linear(64, 1),
                ).to(self.device)
                self.optim = optim.Adam(
                    [
                        {"params": self.actor.parameters()},
                        {"params": self.critic.parameters()},
                        {"params": [self.log_std]},
                    ],
                    lr=getattr(self, "actor_lr", 3e-4),
                    betas=(0.9, 0.999),
                    eps=1e-8,
                )
            else:
                self.actor = nn.Sequential(
                    nn.Linear(self.n_features, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, self.n_actions),
                ).to(self.device)
                self.critic = nn.Sequential(
                    nn.Linear(self.n_features, 64),
                    nn.ReLU(),
                    nn.Linear(64, 64),
                    nn.ReLU(),
                    nn.Linear(64, 1),
                ).to(self.device)
                self.optim = optim.Adam(
                    [{"params": list(self.actor.parameters()) + list(self.critic.parameters())}],
                    lr=getattr(self, "actor_lr", 3e-4),
                    betas=(0.9, 0.999),
                    eps=1e-8,
                )

    def select_action(self, state):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        value, logits = self.forward(state)

        if self.is_continuous_actions:
            mean = logits
            std = self.log_std.exp()
            dist = torch.distributions.Normal(mean, std)

            raw_action = dist.rsample()
            action = torch.tanh(raw_action)

            gaussian_logp = dist.log_prob(raw_action).sum(-1)
            squash_correction = torch.log(1 - action.pow(2) + 1e-6).sum(-1)
            log_prob = gaussian_logp - squash_correction

            entropy = dist.entropy().sum(-1)

            return action, log_prob, value, entropy, raw_action

        else:
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()

            return action, log_prob, value, entropy

    def update_agent(self, rollouts, hp):
        if self.atari_mode:
            T, N, C, H, W = rollouts["states"].shape
            obs_dim = C * H * W
        else:
            T, N, obs_dim = rollouts["states"].shape
        device = rollouts["states"].device

        advantages = torch.zeros(T, N, device=device)
        gae = 0.0
        for t in reversed(range(T - 1)):
            td_error = (
                rollouts["rewards"][t]
                + hp["gamma"] * rollouts["value_preds"][t + 1] * rollouts["masks"][t]
                - rollouts["value_preds"][t]
            )
            gae = td_error + hp["gamma"] * hp["lam"] * rollouts["masks"][t] * gae
            advantages[t] = gae

        returns = advantages + rollouts["value_preds"]

        if self.atari_mode:
            states_flat = rollouts["states"].reshape(-1, hp["stack_size"], 84, 84)
        else:
            states_flat = rollouts["states"].reshape(-1, obs_dim)

        if self.is_continuous_actions:
            actions_flat = rollouts["actions"].reshape(-1, self.n_actions)
        else:
            actions_flat = rollouts["actions"].reshape(-1)

        returns_flat = returns.reshape(-1).detach()
        old_log_probs_flat = rollouts["old_log_probs"].reshape(-1).detach()
        advantages_flat = advantages.reshape(-1).detach()

        total_size = states_flat.size(0)
        actor_loss_epoch = 0
        critic_loss_epoch = 0
        entropy_epoch = 0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(total_size)
            for start in range(0, total_size, self.ppo_batch_size):
                idx = perm[start : start + self.ppo_batch_size]
                batch_states = states_flat[idx]
                batch_actions = actions_flat[idx]
                batch_returns = returns_flat[idx]
                batch_adv = advantages_flat[idx]
                batch_old_log_probs = old_log_probs_flat[idx]

                new_values, new_logits = self.forward(batch_states)

                if self.is_continuous_actions:
                    mean = new_logits
                    std = self.log_std.exp()
                    dist = torch.distributions.Normal(mean, std)

                    raw_actions = batch_actions
                    if raw_actions.ndim == 1:
                        raw_actions = raw_actions.unsqueeze(-1)

                    gaussian_logp = dist.log_prob(raw_actions).sum(-1)
                    squash_correction = torch.log(1 - torch.tanh(raw_actions).pow(2) + 1e-6).sum(-1)
                    new_log_probs = gaussian_logp - squash_correction

                    entropy = dist.entropy().sum(-1).mean()

                else:
                    dist = torch.distributions.Categorical(logits=new_logits)
                    new_log_probs = dist.log_prob(batch_actions)
                    entropy = dist.entropy().mean()

                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * batch_adv
                actor_loss = -torch.min(surr1, surr2).mean() - hp["ent_coef"] * entropy
                critic_loss = (batch_returns - new_values.squeeze(-1)).pow(2).mean()
                critic_loss = critic_loss * hp["vf_coef"]

                self.update_parameters(critic_loss, actor_loss)

                actor_loss_epoch += actor_loss.item()
                critic_loss_epoch += critic_loss.item()
                entropy_epoch += entropy.item()

        num_updates = (total_size // self.ppo_batch_size) * self.ppo_epochs
        return critic_loss_epoch / num_updates, actor_loss_epoch / num_updates, entropy_epoch / num_updates


class EpisodeBuffer:

    def __init__(self, n_envs, device):
        self.n_envs = n_envs
        self.device = device
        self._staging = [
            {
                "states": [],
                "actions": [],
                "old_log_probs": [],
                "old_logits": [],
                "old_stds": [],
                "pre_tanh_actions": [],
                "rewards": [],
            }
            for _ in range(n_envs)
        ]

    def add_step(
        self,
        env_idx,
        state,
        action,
        old_log_prob,
        old_logits,
        reward,
        pre_tanh_action=None,
        old_std=None,
    ):
        s = state if torch.is_tensor(state) else torch.as_tensor(state, dtype=torch.float32, device=self.device)
        self._staging[env_idx]["states"].append(s)

        a = action if torch.is_tensor(action) else torch.as_tensor(action, device=self.device)
        self._staging[env_idx]["actions"].append(a)

        lp = old_log_prob if torch.is_tensor(old_log_prob) else torch.as_tensor(old_log_prob, device=self.device)
        self._staging[env_idx]["old_log_probs"].append(lp)

        logits = (
            old_logits
            if torch.is_tensor(old_logits)
            else torch.as_tensor(old_logits, dtype=torch.float32, device=self.device)
        )

        self._staging[env_idx]["old_logits"].append(logits)

        if old_std is not None:
            std_tensor = old_std if torch.is_tensor(old_std) else torch.as_tensor(old_std, device=self.device)
            self._staging[env_idx]["old_stds"].append(std_tensor)
        else:
            self._staging[env_idx]["old_stds"].append(None)

        if pre_tanh_action is not None:
            pre_tanh_tensor = (
                pre_tanh_action
                if torch.is_tensor(pre_tanh_action)
                else torch.as_tensor(pre_tanh_action, device=self.device)
            )
            self._staging[env_idx]["pre_tanh_actions"].append(pre_tanh_tensor)
        else:
            self._staging[env_idx]["pre_tanh_actions"].append(None)

        r = reward if torch.is_tensor(reward) else torch.as_tensor(reward, dtype=torch.float32, device=self.device)
        self._staging[env_idx]["rewards"].append(r)

    def end_episode(self, env_idx):
        stage = self._staging[env_idx]
        if len(stage["states"]) == 0:
            return None

        states = torch.stack(stage["states"], dim=0)
        actions = torch.stack(stage["actions"], dim=0).view(-1)
        old_log_probs = torch.stack(stage["old_log_probs"], dim=0).view(-1)
        old_logits = torch.stack(stage["old_logits"], dim=0)
        rewards = torch.stack(stage["rewards"], dim=0).view(-1)

        old_stds = torch.stack(stage["old_stds"], dim=0) if stage["old_stds"][0] is not None else None
        pre_tanh_actions = (
            torch.stack(stage["pre_tanh_actions"], dim=0) if stage["pre_tanh_actions"][0] is not None else None
        )

        episode = {
            "states": states,
            "actions": actions,
            "old_log_probs": old_log_probs,
            "old_logits": old_logits,
            "old_stds": old_stds,
            "pre_tanh_actions": pre_tanh_actions,
            "rewards": rewards,
        }

        self._staging[env_idx] = {
            "states": [],
            "actions": [],
            "old_log_probs": [],
            "old_logits": [],
            "old_stds": [],
            "pre_tanh_actions": [],
            "rewards": [],
        }

        return episode

    def reset_all(self):
        self._staging = [
            {
                "states": [],
                "actions": [],
                "old_log_probs": [],
                "old_logits": [],
                "old_stds": [],
                "pre_tanh_actions": [],
                "rewards": [],
            }
            for _ in range(self.n_envs)
        ]


class GRPO(A2CBase):
    def __init__(
        self,
        *args,
        ppo_epochs=None,
        ppo_batch_size=None,
        clip_coef=None,
        adv_clip=None,
        use_kl=None,
        is_continuous_actions=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if ppo_epochs is None or ppo_batch_size is None or clip_coef is None:
            raise ValueError("ppo_epochs, ppo_batch_size and clip_coef must be provided for GRPO")
        self.ppo_epochs = ppo_epochs
        self.ppo_batch_size = ppo_batch_size
        self.clip_coef = clip_coef
        self.adv_clip = adv_clip
        self.use_kl = use_kl
        self.is_continuous_actions = is_continuous_actions

        if self.atari_mode:
            self.conv1 = nn.Conv2d(in_channels=self.stack_size, out_channels=16, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2)

            self.flattened_size = 32 * 9 * 9
            self.flatten = nn.Flatten()

            self.fc = nn.Sequential(nn.Linear(self.flattened_size, 256), nn.ReLU()).to(self.device)

            self.actor = nn.Linear(256, self.n_actions).to(self.device)
            self.feature_extractor_params = list(self.conv1.parameters()) + list(self.conv2.parameters())

            self.optim = optim.RMSprop(
                [
                    {"params": self.conv1.parameters()},
                    {"params": self.conv2.parameters()},
                    {"params": self.fc.parameters()},
                    {"params": self.actor.parameters()},
                ],
                lr=self.actor_lr,
                alpha=0.99,
                eps=1e-5,
            )

        elif is_continuous_actions:
            self.actor = nn.Sequential(
                nn.Linear(self.n_features, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, self.n_actions),
                nn.Tanh(),
            ).to(self.device)
            self.log_std = nn.Parameter(torch.zeros(self.n_actions, device=self.device))

            self.optim = optim.Adam(
                [
                    {"params": self.actor.parameters()},
                    {"params": [self.log_std]},
                ],
                lr=self.actor_lr,
                betas=(0.9, 0.999),
                eps=1e-8,
            )

        else:
            self.actor = nn.Sequential(
                nn.Linear(self.n_features, 64),
                nn.ReLU(),
                nn.Linear(64, 64),
                nn.ReLU(),
                nn.Linear(self.n_features, self.n_actions),
            ).to(self.device)

            self.optim = optim.Adam(
                [{"params": self.actor.parameters(), "lr": self.actor_lr}],
                betas=(0.9, 0.999),
                eps=1e-8,
                weight_decay=0,
            )

    def forward(self, x):
        x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        if self.atari_mode:
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.flatten(x)
            x = self.fc(x)
            out = self.actor(x)
        else:
            out = self.actor(x)
        return out

    def select_action(self, x):
        out = self.forward(x)

        if self.is_continuous_actions:
            mu = out
            std = torch.exp(self.log_std)
            dist = torch.distributions.Normal(mu, std)
            action = dist.rsample()
            self._pre_tanh_action = action
            action_tanh = torch.tanh(action)
            log_prob = dist.log_prob(action).sum(dim=-1) - torch.log(1 - action_tanh.pow(2) + 1e-6).sum(dim=-1)
            entropy = dist.entropy().sum(dim=-1)
            return action_tanh, log_prob, mu, entropy

        else:
            logits = out
            dist = torch.distributions.Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            entropy = dist.entropy()
            return action, log_prob, logits, entropy

    def update_agent(self, episodes, hp):
        if len(episodes) == 0:
            return 0.0, 0.0, 0.0

        device = self.device

        if hp["grpo_mc"]:
            returns = torch.tensor([ep["rewards"].sum() for ep in episodes], device=device)
            baseline = returns.mean()
            for ep in episodes:
                rew = ep["rewards"]
                G_t = torch.flip(torch.cumsum(torch.flip(rew, dims=[0]), dim=0), dims=[0])
                ep["advantages"] = (G_t - baseline).detach()

            all_adv = torch.cat([ep["advantages"] for ep in episodes], dim=0)
            adv_mean = all_adv.mean()
            adv_std = all_adv.std(unbiased=False)
            if adv_std < 1e-8:
                adv_std = 1.0
            for ep in episodes:
                ep["advantages"] = ((ep["advantages"] - adv_mean) / (adv_std + 1e-8)).detach()
        else:
            returns = torch.tensor([ep["rewards"].sum() for ep in episodes], dtype=torch.float32, device=device)
            ret_mean = returns.mean()
            ret_std = returns.std(unbiased=False)
            advantages_episode = (
                torch.zeros_like(returns) if ret_std < 1e-8 else (returns - ret_mean) / (ret_std + 1e-8)
            )
            advantages_episode = advantages_episode.detach()
            for i, ep in enumerate(episodes):
                L = ep["rewards"].shape[0]
                ep["advantages"] = torch.full((L,), advantages_episode[i].item(), dtype=torch.float32, device=device)

        states_cat = torch.cat([ep["states"] for ep in episodes], dim=0)
        if self.is_continuous_actions:
            actions_cat = torch.cat([ep["pre_tanh_actions"].view(-1, self.n_actions) for ep in episodes], dim=0)
            old_stds_cat = torch.cat([ep["old_stds"].view(-1, self.n_actions) for ep in episodes], dim=0)
        else:
            actions_cat = torch.cat([ep["actions"].view(-1) for ep in episodes], dim=0)
            old_stds_cat = None

        old_logp_cat = torch.cat([ep["old_log_probs"].view(-1) for ep in episodes], dim=0)
        old_logits_cat = torch.cat([ep["old_logits"] for ep in episodes], dim=0)
        adv_cat = torch.cat([ep["advantages"].view(-1) for ep in episodes], dim=0)
        if self.adv_clip is not None:
            adv_cat = torch.clamp(adv_cat, -self.adv_clip, self.adv_clip)
        adv_cat = adv_cat.detach()

        total_size = states_cat.shape[0]
        print(f"Total size: {total_size}")

        actor_loss_epoch = 0.0
        entropy_epoch = 0.0
        num_updates = 0

        for _ in range(self.ppo_epochs):
            perm = torch.randperm(total_size, device=device)
            for start in range(0, total_size, self.ppo_batch_size):
                idx = perm[start : start + self.ppo_batch_size]
                mb_states = states_cat[idx]
                mb_actions = actions_cat[idx] if self.is_continuous_actions else actions_cat[idx]
                mb_adv = adv_cat[idx]
                mb_old_logp = old_logp_cat[idx]
                mb_old_logits = old_logits_cat[idx]
                mb_old_stds = old_stds_cat[idx] if self.is_continuous_actions else None

                logits_or_mu = self.forward(mb_states)

                if self.is_continuous_actions:
                    mu = logits_or_mu
                    std = torch.exp(self.log_std).expand_as(mu)
                    dist = torch.distributions.Normal(mu, std)

                    if mb_actions.ndim == 1:
                        mb_actions = mb_actions.unsqueeze(-1)
                    new_logp = dist.log_prob(mb_actions).sum(dim=-1) - torch.log(
                        1 - torch.tanh(mb_actions).pow(2) + 1e-6
                    ).sum(dim=-1)
                    entropy = dist.entropy().sum(dim=-1).mean()
                    ratio = torch.exp(new_logp - mb_old_logp)

                else:
                    dist = torch.distributions.Categorical(logits=logits_or_mu)
                    new_logp = dist.log_prob(mb_actions)
                    entropy = dist.entropy().mean()
                    ratio = torch.exp(new_logp - mb_old_logp)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef) * mb_adv

                if self.use_kl:
                    if self.is_continuous_actions:
                        old_dist = torch.distributions.Normal(mb_old_logits, mb_old_stds)
                        kl = torch.distributions.kl_divergence(dist, old_dist).sum(dim=-1).mean()
                        actor_loss = -torch.min(surr1, surr2).mean() - hp["ent_coef"] * entropy + hp["kl_coef"] * kl
                    else:
                        old_dist = torch.distributions.Categorical(logits=mb_old_logits)
                        kl = torch.distributions.kl_divergence(dist, old_dist).mean()
                        actor_loss = -torch.min(surr1, surr2).mean() - hp["ent_coef"] * entropy + hp["kl_coef"] * kl
                else:
                    actor_loss = -torch.min(surr1, surr2).mean() - hp["ent_coef"] * entropy

                self.optim.zero_grad()
                actor_loss.backward()
                self.optim.step()

                actor_loss_epoch += float(actor_loss.item())
                entropy_epoch += float(entropy.item())
                num_updates += 1

        return 0.0, actor_loss_epoch / num_updates, entropy_epoch / num_updates


def A2C(agent_type, **kwargs):
    if agent_type == "a2c":
        return A2CSimple(**kwargs)
    elif agent_type == "ppo":
        return PPO(**kwargs)
    elif agent_type == "grpo":
        return GRPO(**kwargs)
    else:
        raise ValueError(f"Unknown agent_type {agent_type}")
