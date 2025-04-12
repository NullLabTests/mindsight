#!/usr/bin/env python3
import os
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import logging
from datetime import datetime

# Configure logging for production-quality output.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)

# ==========================
# Environment Module
# ==========================
class GridWorld:
    """A simple 1D grid environment."""
    def __init__(self, size=10, goal=9):
        self.size = size
        self.goal = goal
        self.position = 0

    def reset(self):
        self.position = 0
        return self.position

    def step(self, action):
        """Moves the agent and returns (next_state, reward, done)."""
        if action == 0:  # Move left
            self.position = max(0, self.position - 1)
        elif action == 2:  # Move right
            self.position = min(self.size - 1, self.position + 1)
        # Action 1 is 'stay'
        reward = 1.0 if self.position == self.goal else -0.1
        done = self.position == self.goal
        return self.position, reward, done

# ==========================
# Agent Module
# ==========================
class SelfAwareAgent(nn.Module):
    """A self-modeling RL agent with an RNN backend."""
    def __init__(self):
        super().__init__()
        # Input (state=1, previous action=3) -> 4; hidden state size = 16.
        self.rnn = nn.RNN(4, 16, batch_first=True)
        self.action_head = nn.Linear(16, 3)      # To choose the next action.
        self.self_pred_head = nn.Linear(16, 3)     # To predict the next action.

    def forward(self, state, prev_action, hidden):
        # Concatenate state and previous action along the feature dimension.
        x = torch.cat((state, prev_action), dim=-1)
        output, hidden = self.rnn(x, hidden)
        # Generate action probabilities and self prediction probabilities.
        action_logits = self.action_head(output)
        self_pred_logits = self.self_pred_head(output)
        action_probs = torch.softmax(action_logits, dim=-1)
        self_pred = torch.softmax(self_pred_logits, dim=-1)
        return action_probs, self_pred, hidden

# ==========================
# Trainer Module
# ==========================
class Trainer:
    """Trainer class to organize training, checkpointing, and visualization."""
    def __init__(self, config):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logging.info("Using device: %s", self.device)

        # Create environment and agent.
        self.env = GridWorld(size=config.get("grid_size", 10),
                             goal=config.get("goal", 9))
        self.agent = SelfAwareAgent().to(self.device)
        self.optimizer = optim.Adam(self.agent.parameters(), lr=config.get("learning_rate", 0.001))
        self.checkpoint_file = config.get("checkpoint_file", "model_checkpoint.pth")
        self.episode = 0

        self._load_checkpoint()

        # Set up visualization
        self.episode_rewards = []
        self.episode_accuracies = []
        self._setup_visualization()

    def _load_checkpoint(self):
        if os.path.exists(self.checkpoint_file):
            checkpoint = torch.load(self.checkpoint_file, map_location=self.device)
            self.agent.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.episode = checkpoint["episode"] + 1
            logging.info("Loaded checkpoint from episode %d", self.episode)
        else:
            logging.info("No checkpoint found. Starting fresh training.")

    def _save_checkpoint(self):
        checkpoint = {
            "episode": self.episode,
            "model_state_dict": self.agent.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict()
        }
        torch.save(checkpoint, self.checkpoint_file)
        logging.info("Checkpoint saved at episode %d", self.episode)

    def _setup_visualization(self):
        plt.ion()  # Interactive plotting
        self.fig, (self.ax_reward, self.ax_acc) = plt.subplots(2, 1, figsize=(8, 8))
        self.ax_reward.set_title("Total Reward per Episode")
        self.ax_reward.set_xlabel("Episode")
        self.ax_reward.set_ylabel("Reward")
        self.ax_acc.set_title("Self-Prediction Accuracy")
        self.ax_acc.set_xlabel("Episode")
        self.ax_acc.set_ylabel("Accuracy")

    def _update_visualization(self):
        self.ax_reward.clear()
        self.ax_acc.clear()
        self.ax_reward.plot(self.episode_rewards, label="Total Reward")
        self.ax_reward.set_xlabel("Episode")
        self.ax_reward.set_ylabel("Reward")
        self.ax_reward.legend()
        self.ax_acc.plot(self.episode_accuracies, label="Self-Prediction Accuracy", color="orange")
        self.ax_acc.set_xlabel("Episode")
        self.ax_acc.set_ylabel("Accuracy")
        self.ax_acc.legend()
        plt.tight_layout()
        plt.pause(0.001)

    def _run_episode(self):
        # Reset environment and create initial tensors.
        state = self.env.reset()
        hidden = torch.zeros(1, 1, 16, device=self.device)  # RNN initial hidden state.
        prev_action = torch.zeros(1, 1, 3, device=self.device)  # One-hot vector for previous action.
        done = False

        rewards = []
        actions = []
        action_prob_list = []
        self_preds = []

        while not done:
            # Prepare state tensor: shape (1, 1, 1).
            state_tensor = torch.tensor([[[state]]], dtype=torch.float32, device=self.device)
            action_probs, self_pred, hidden = self.agent(state_tensor, prev_action, hidden)
            # Sample an action from the distribution.
            action = torch.multinomial(action_probs[0, 0], 1).item()
            next_state, reward, done = self.env.step(action)

            rewards.append(reward)
            actions.append(action)
            action_prob_list.append(action_probs)
            self_preds.append(self_pred)

            state = next_state
            # Reset previous action tensor and set the chosen action.
            prev_action = torch.zeros(1, 1, 3, device=self.device)
            prev_action[0, 0, action] = 1

        return rewards, actions, action_prob_list, self_preds

    def _compute_loss(self, rewards, actions, action_prob_list, self_preds):
        """Computes combined loss from action selection and self-prediction."""
        returns = []
        R = 0
        # Discounted return calculation.
        for r in reversed(rewards):
            R = r + self.config.get("discount_factor", 0.99) * R
            returns.insert(0, R)
        returns = torch.tensor(returns, device=self.device)

        loss = 0
        for t in range(len(rewards)):
            log_prob = torch.log(action_prob_list[t][0, 0, actions[t]] + 1e-10)
            loss += -log_prob * returns[t]

            if t < len(rewards) - 1:
                next_action_onehot = torch.zeros(3, device=self.device)
                next_action_onehot[actions[t + 1]] = 1
                self_pred_loss = -torch.sum(next_action_onehot * torch.log(self_preds[t][0, 0] + 1e-10))
                loss += self.config.get("self_pred_weight", 0.1) * self_pred_loss

        return loss

    def _calculate_accuracy(self, actions, self_preds):
        """Calculates self-prediction accuracy (comparing prediction to next step action)."""
        correct = sum(torch.argmax(p[0, 0]).item() == a for p, a in zip(self_preds[:-1], actions[1:]))
        accuracy = correct / max(1, len(actions) - 1)
        return accuracy

    def train(self):
        logging.info("Starting training loop. Press Ctrl+C to stop.")
        try:
            while True:
                # Run one complete episode.
                rewards, actions, action_prob_list, self_preds = self._run_episode()
                loss = self._compute_loss(rewards, actions, action_prob_list, self_preds)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                accuracy = self._calculate_accuracy(actions, self_preds)
                total_reward = sum(rewards)

                logging.info("Episode %d: Total Reward = %.2f, Self-Prediction Accuracy = %.2f, Loss = %.4f",
                             self.episode, total_reward, accuracy, loss.item())

                # Update visualization data.
                self.episode_rewards.append(total_reward)
                self.episode_accuracies.append(accuracy)
                self._update_visualization()

                # Save periodic checkpoints.
                if self.episode % self.config.get("checkpoint_interval", 10) == 0:
                    self._save_checkpoint()

                self.episode += 1

        except KeyboardInterrupt:
            logging.info("Training interrupted. Saving final checkpoint...")
            self._save_checkpoint()
            plt.ioff()
            plt.show()

# ==========================
# Main Entry Point
# ==========================
if __name__ == '__main__':
    # Production configuration parameters.
    config = {
        "grid_size": 10,
        "goal": 9,
        "learning_rate": 0.001,
        "discount_factor": 0.99,
        "self_pred_weight": 0.1,
        "checkpoint_interval": 10,
        "checkpoint_file": "model_checkpoint.pth",
    }

    trainer = Trainer(config)
    trainer.train()

