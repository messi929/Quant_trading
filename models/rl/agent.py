"""PPO Agent for sector allocation decisions."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic(nn.Module):
    """Shared backbone with separate actor and critic heads.

    Actor outputs continuous sector allocations.
    Critic estimates state value for advantage computation.
    """

    def __init__(
        self,
        state_dim: int,
        n_sectors: int = 11,
        hidden_dims: list[int] = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [256, 128]

        # Shared feature extractor
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.LayerNorm(h_dim),
                nn.Tanh(),
            ])
            in_dim = h_dim
        self.backbone = nn.Sequential(*layers)

        # Actor: mean and log_std for continuous actions
        self.actor_mean = nn.Linear(hidden_dims[-1], n_sectors)
        self.actor_log_std = nn.Parameter(torch.zeros(n_sectors))

        # Critic: state value
        self.critic = nn.Linear(hidden_dims[-1], 1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)

    def forward(self, state: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        """
        Args:
            state: (batch, state_dim) observation

        Returns:
            action_dist: Normal distribution over actions
            value: (batch, 1) state value estimate
        """
        features = self.backbone(state)
        action_mean = torch.tanh(self.actor_mean(features))
        action_std = torch.exp(self.actor_log_std.clamp(-5, 2))
        action_dist = Normal(action_mean, action_std)
        value = self.critic(features)
        return action_dist, value

    def get_action(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample action and compute log probability.

        Returns:
            action, log_prob, value
        """
        dist, value = self.forward(state)
        if deterministic:
            action = dist.mean
        else:
            action = dist.sample()
        action = torch.tanh(action)  # Bound to [-1, 1]
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob, value.squeeze(-1)

    def evaluate_actions(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update.

        Returns:
            log_prob, value, entropy
        """
        dist, value = self.forward(states)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, value.squeeze(-1), entropy


class PPOAgent:
    """Proximal Policy Optimization agent for sector trading."""

    def __init__(
        self,
        state_dim: int,
        n_sectors: int = 11,
        hidden_dims: list[int] = None,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        n_epochs: int = 10,
        batch_size: int = 64,
        device: torch.device = None,
    ):
        self.device = device or torch.device("cpu")
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size

        self.policy = ActorCritic(
            state_dim, n_sectors, hidden_dims
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.policy.parameters(), lr=learning_rate, eps=1e-5
        )

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
    ) -> tuple[np.ndarray, float, float]:
        """Select action for environment step.

        Returns:
            action, log_prob, value
        """
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            action, log_prob, value = self.policy.get_action(
                state_t, deterministic
            )
        return (
            action.squeeze(0).cpu().numpy(),
            log_prob.item(),
            value.item(),
        )

    def compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[bool],
        next_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.

        Returns:
            advantages, returns
        """
        advantages = np.zeros(len(rewards), dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_val = next_value
            else:
                next_val = values[t + 1]

            next_non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_val * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae

        returns = advantages + np.array(values, dtype=np.float32)
        return advantages, returns

    def update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
    ) -> dict[str, float]:
        """PPO update step.

        Returns:
            Dict with policy_loss, value_loss, entropy, total_loss
        """
        states_t = torch.FloatTensor(states).to(self.device)
        actions_t = torch.FloatTensor(actions).to(self.device)
        old_log_probs_t = torch.FloatTensor(old_log_probs).to(self.device)
        advantages_t = torch.FloatTensor(advantages).to(self.device)
        returns_t = torch.FloatTensor(returns).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (
            advantages_t.std() + 1e-8
        )

        total_policy_loss = 0.0
        total_value_loss = 0.0
        total_entropy = 0.0
        n_updates = 0

        n_samples = len(states)
        for _ in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]

                log_probs, values, entropy = self.policy.evaluate_actions(
                    states_t[batch_idx],
                    actions_t[batch_idx],
                )

                # PPO clipped objective
                ratio = torch.exp(log_probs - old_log_probs_t[batch_idx])
                surr1 = ratio * advantages_t[batch_idx]
                surr2 = (
                    torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon)
                    * advantages_t[batch_idx]
                )
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_loss = nn.functional.mse_loss(values, returns_t[batch_idx])

                # Entropy bonus
                entropy_loss = -entropy.mean()

                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    + self.entropy_coef * entropy_loss
                )

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                nn.utils.clip_grad_norm_(
                    self.policy.parameters(), self.max_grad_norm
                )
                self.optimizer.step()

                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
                n_updates += 1

        return {
            "policy_loss": total_policy_loss / max(1, n_updates),
            "value_loss": total_value_loss / max(1, n_updates),
            "entropy": total_entropy / max(1, n_updates),
        }
