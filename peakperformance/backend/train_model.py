"""
Module for training a reinforcement learning model for contract negotiations.

This module loads player data, defines the Deep Q-Network (DQN) model, a contract
negotiation environment, and a reinforcement learning agent. It then trains the RL model
using the provided player data and saves both the trained model and a reward progression plot.

Author: Balaji Boopal
Date: March 16, 2025

Global Variables:
    None

Returns:
    None
"""

from pathlib import Path
import random
import collections
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
import matplotlib.pyplot as plt


def load_player_data(file_path: str) -> pd.DataFrame:
    """
    Loads player data from a CSV file.

    Parameters:
        file_path (str): The path to the CSV file containing player data.

    Returns:
        pd.DataFrame: DataFrame containing player data with rows having missing values
                      in "GROSS P/W (EUR)", "Rating", "CLUB", or "pos" dropped.
    """
    player_data = pd.read_csv(file_path)
    return player_data.dropna(subset=["GROSS P/W (EUR)", "Rating", "CLUB", "pos"])


class DeepQNetwork(nn.Module):
    """
    Deep Q-Network (DQN) model for contract negotiations.

    Parameters:
        state_dim (int): The dimension of the state input.
        action_dim (int): The number of possible actions.
    """

    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: The output Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class ContractNegotiationEnvironment:
    """
    Environment simulating player contract negotiations.

    Parameters:
        player_data (pd.DataFrame): DataFrame containing player data.

    Attributes:
        action_space (list): List of possible actions [Accept, Negotiate, Change Length, Reject].
        state (np.ndarray): The current state of the environment.
        done (bool): Flag indicating whether the negotiation is finished.
        player (pd.Series): The current player's data.
    """

    def __init__(self, player_data: pd.DataFrame):
        self.player_data = player_data
        self.state = None
        self.done = False
        self.action_space = [0, 1, 2, 3]
        self.player = None

    def reset(self, player_name: str) -> np.ndarray:
        """
        Resets the environment for a specific player.

        Parameters:
            player_name (str): The name of the player for the negotiation.

        Returns:
            np.ndarray: The initial state for the player.
        """
        player_seasons = self.player_data[self.player_data["PLAYER"]
                                          == player_name]
        self.player = player_seasons.iloc[-1]
        current_wage = self.player["GROSS P/W (EUR)"]

        self.state = np.array([
            self.player["age"],
            self.player["Rating"],
            current_wage,
            0,  # Proposed Wage
            3   # Default contract length
        ], dtype=float)
        self.done = False
        return self.state

    def step(self, action: int, proposed_wage: float, contract_length: int) -> tuple:
        """
        Takes an action in the contract negotiation.

        Parameters:
            action (int): The action to perform.
            proposed_wage (float): The wage proposed by the club.
            contract_length (int): The proposed contract length in years.

        Returns:
            tuple: A tuple containing:
                - np.ndarray: The current state.
                - float: The reward obtained.
                - str or None: A counteroffer message if applicable.
                - list: A log of negotiation events.
                - bool: A flag indicating whether negotiation is done.
        """
        age, rating, current_wage, _, _ = self.state
        expected_wage = current_wage * (1.5 + 0.1 * (rating - 6.5))
        if age > 30:
            expected_wage *= 0.9

        base_reward = (rating - 5) * 1.2
        age_penalty = max(0, (age - 30) * -0.5)

        counteroffer = None
        negotiation_log = []

        if proposed_wage < expected_wage * 0.8:
            reward = -50
            self.done = True
            counteroffer = (
                f"Player refused. Minimum acceptable offer is €{int(expected_wage * 0.9):,}/week!"
            )
            negotiation_log.append("Player outright rejected the low offer.")
            return (
                np.array(self.state, dtype=float),
                reward,
                counteroffer,
                negotiation_log,
                self.done
            )

        if action == 0:  # Accept Contract
            if proposed_wage >= expected_wage:
                reward = 100 + (proposed_wage / expected_wage) * 50
                self.done = True
                negotiation_log.append(
                    f"Contract Accepted at €{int(proposed_wage):,}/week for"
                    "{contract_length} years."
                )
            else:
                reward = -20
                self.done = False

        elif action == 1:  # Negotiate Higher Wage
            if proposed_wage >= expected_wage:
                reward = 30 + base_reward + age_penalty
                counteroffer = "Player has accepted the new wage offer!"
                self.done = True
            else:
                reward = 10 + base_reward
                counteroffer = f"Player wants at least €{int(expected_wage):,}/week!"
                negotiation_log.append(
                    f"Player countered with €{int(expected_wage):,}/week")

        elif action == 2:  # Change Contract Length
            reward = 8 if abs(contract_length - 4) <= 1 else -5
            counteroffer = "Player prefers a 4-year contract instead!"

        else:  # Reject Offer
            reward = -10
            counteroffer = (
                f"Player rejected the offer. Expected €{int(expected_wage):,}/week for 4 years."
            )
            self.done = True
            negotiation_log.append("Player rejected the contract offer.")

        return (
            np.array(self.state, dtype=float),
            reward,
            counteroffer,
            negotiation_log,
            self.done
        )


class ReinforcementLearningAgent:
    """
    Reinforcement Learning agent for contract negotiation of salaries.

    Parameters:
        state_dim (int): Dimension of the state space.
        action_dim (int): Number of possible actions.
        learning_rate (float, optional): Learning rate for the optimizer.
        gamma (float, optional): Discount factor for future rewards.

    Attributes:
        model (DeepQNetwork): The DQN model.
        optimizer (torch.optim.Optimizer): Optimizer for training the model.
        gamma (float): Discount factor.
        epsilon (float): Exploration rate.
        epsilon_decay (float): Decay rate for epsilon.
        epsilon_min (float): Minimum exploration rate.
        memory (collections.deque): Replay memory.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        learning_rate: float = 0.0005,
        gamma: float = 0.95
    ):
        self.model = DeepQNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = 1.0
        self.epsilon_decay = 0.9998
        self.epsilon_min = 0.1
        self.memory = collections.deque(maxlen=10000)

    def select_action(self, state: np.ndarray) -> int:
        """
        Selects an action using an epsilon-greedy policy.

        Parameters:
            state (np.ndarray): The current state of the environment.

        Returns:
            int: The chosen action.
        """
        if random.uniform(0, 1) < self.epsilon:
            return random.choice([0, 1, 2, 3])
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        return torch.argmax(self.model(state_tensor)).item()

    def save_model(self, path: str = None, use_jit: bool = False) -> None:
        """
        Saves the RL model to a file.

        Parameters:
            path (str, optional): The file path to save the model. 
            Defaults to assets/rl_contract_model.pth.
            use_jit (bool, optional): If True, saves the model using TorchScript (JIT);
                                      otherwise, saves the state_dict.

        Returns:
            None
        """
        if path is None:
            path = Path.cwd() / "assets" / "rl_contract_model.pth"

        if use_jit:
            scripted_model = torch.jit.script(self.model)
            scripted_model.save(path)
            print("🟢 Saved model using TorchScript (JIT).")
        else:
            torch.save(self.model.state_dict(), path)
            print("🟡 Saved model using `state_dict` format.")

    def load_model(self, path: str = None) -> None:
        """
        Loads the RL model from a file.

        Parameters:
            path (str, optional): The file path from which to load the model. 
            Defaults to assets/rl_contract_model.pth.

        Returns:
            None
        """
        if path is None:
            path = Path.cwd() / "assets" / "rl_contract_model.pth"

        try:
            self.model.load_state_dict(torch.load(
                path, map_location=torch.device("cpu")))
            print("✅ Loaded model using `state_dict` format.")
        except (RuntimeError, TypeError, ValueError) as e:
            print(f"⚠️ Detected TorchScript model, error: {e}")
            self.model = torch.jit.load(path, map_location=torch.device("cpu"))
            print("✅ Loaded model using TorchScript (JIT).")


def train_model(player_data: pd.DataFrame) -> None:
    """
    Trains the RL model and saves both the trained model and a reward progression plot.

    Parameters:
        player_data (pd.DataFrame): DataFrame containing player data used for training.

    Returns:
        None
    """
    env = ContractNegotiationEnvironment(player_data)
    agent = ReinforcementLearningAgent(state_dim=5, action_dim=4)
    reward_history = []

    for _ in range(3000):
        player_name = random.choice(player_data["PLAYER"].unique())
        state = env.reset(player_name)
        done_flag = False
        total_reward = 0

        while not done_flag:
            act = agent.select_action(state)
            wage = state[2] * (1 + random.uniform(-0.1, 0.2))
            length = random.randint(2, 5)
            next_state, reward, _, _, done_flag = env.step(act, wage, length)
            agent.memory.append((state, act, reward, next_state))
            state = next_state
            total_reward += reward

        reward_history.append(total_reward)

    model_save_path = Path.cwd() / "assets" / "rl_contract_model.pth"
    torch.save(agent.model.state_dict(), model_save_path)

    plt.figure(figsize=(10, 5))
    plt.plot(reward_history)
    plt.xlabel("Episodes")
    plt.ylabel("Total Reward")
    plt.title("Q-Learning Reward Progression")
    plt.savefig("reward_plot.png")
    plt.close()

    print("RL Model Trained & Saved!")


def main() -> None:
    """
    Main function to load player data, train the RL model, and save the model and reward plot.

    Returns:
        None
    """
    data_file_path = (
        Path.cwd() / "dataset" / "Ratings Combined" /
        "filtered_playerratingssalaries.csv"
    )
    loaded_player_data = load_player_data(data_file_path)
    train_model(loaded_player_data)


if __name__ == "__main__":
    main()
