"""RL training loop with rollout collection and PPO updates."""

import numpy as np
from loguru import logger

from models.rl.agent import PPOAgent
from models.rl.environment import SectorTradingEnv
from utils.storage import StorageManager


class RLTrainer:
    """Trains the PPO agent on the sector trading environment."""

    def __init__(
        self,
        agent: PPOAgent,
        env: SectorTradingEnv,
        n_steps: int = 2048,
        total_timesteps: int = 1_000_000,
    ):
        self.agent = agent
        self.env = env
        self.n_steps = n_steps
        self.total_timesteps = total_timesteps
        self.storage = StorageManager()

    def collect_rollout(self) -> dict:
        """Collect a rollout of n_steps transitions."""
        states = []
        actions = []
        log_probs = []
        rewards = []
        values = []
        dones = []

        state, info = self.env.reset()

        for _ in range(self.n_steps):
            action, log_prob, value = self.agent.select_action(state)

            next_state, reward, terminated, truncated, info = self.env.step(action)
            done = terminated or truncated

            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            rewards.append(reward)
            values.append(value)
            dones.append(done)

            state = next_state
            if done:
                state, info = self.env.reset()

        # Get value of last state for GAE
        _, _, next_value = self.agent.select_action(state, deterministic=True)

        advantages, returns = self.agent.compute_gae(
            rewards, values, dones, next_value
        )

        return {
            "states": np.array(states, dtype=np.float32),
            "actions": np.array(actions, dtype=np.float32),
            "log_probs": np.array(log_probs, dtype=np.float32),
            "advantages": advantages,
            "returns": returns,
            "rewards": rewards,
            "info": info,
        }

    def train(self) -> dict:
        """Full training loop.

        Returns:
            Training history dict
        """
        n_updates = self.total_timesteps // self.n_steps
        history = {
            "episode_rewards": [],
            "policy_losses": [],
            "value_losses": [],
            "entropies": [],
            "portfolio_values": [],
        }

        logger.info(
            f"Starting RL training: {n_updates} updates, "
            f"{self.n_steps} steps/update, "
            f"{self.total_timesteps} total timesteps"
        )

        best_reward = float("-inf")

        for update in range(1, n_updates + 1):
            # Collect rollout
            rollout = self.collect_rollout()

            # PPO update
            update_metrics = self.agent.update(
                rollout["states"],
                rollout["actions"],
                rollout["log_probs"],
                rollout["advantages"],
                rollout["returns"],
            )

            # Track metrics
            mean_reward = np.mean(rollout["rewards"])
            history["episode_rewards"].append(mean_reward)
            history["policy_losses"].append(update_metrics["policy_loss"])
            history["value_losses"].append(update_metrics["value_loss"])
            history["entropies"].append(update_metrics["entropy"])
            history["portfolio_values"].append(
                rollout["info"]["portfolio_value"]
            )

            if update % 10 == 0:
                total_ret = rollout["info"]["total_return"]
                logger.info(
                    f"Update {update}/{n_updates} | "
                    f"Reward: {mean_reward:.4f} | "
                    f"Return: {total_ret:.2%} | "
                    f"P.Loss: {update_metrics['policy_loss']:.4f} | "
                    f"V.Loss: {update_metrics['value_loss']:.4f} | "
                    f"Entropy: {update_metrics['entropy']:.4f}"
                )

            # Save best model
            if mean_reward > best_reward:
                best_reward = mean_reward
                self.storage.save_model_checkpoint(
                    self.agent.policy.state_dict(),
                    "ppo_agent",
                    update,
                    metrics={
                        "mean_reward": mean_reward,
                        "total_return": rollout["info"]["total_return"],
                    },
                )

        logger.info(f"Training complete. Best reward: {best_reward:.4f}")
        return history

    def evaluate(self, n_episodes: int = 10) -> dict:
        """Evaluate trained agent over multiple episodes."""
        all_returns = []
        all_mdd = []

        for ep in range(n_episodes):
            state, info = self.env.reset()
            done = False
            while not done:
                action, _, _ = self.agent.select_action(state, deterministic=True)
                state, reward, terminated, truncated, info = self.env.step(action)
                done = terminated or truncated

            all_returns.append(info["total_return"])
            all_mdd.append(info["max_drawdown"])

        results = {
            "mean_return": np.mean(all_returns),
            "std_return": np.std(all_returns),
            "mean_mdd": np.mean(all_mdd),
            "best_return": np.max(all_returns),
            "worst_return": np.min(all_returns),
        }

        logger.info(
            f"Evaluation ({n_episodes} episodes): "
            f"Return={results['mean_return']:.2%} ± {results['std_return']:.2%}, "
            f"MDD={results['mean_mdd']:.2%}"
        )
        return results
