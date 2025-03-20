"""
Unit tests for the reinforcement learning training module for contract negotiations.

This module tests all functions and classes defined in the RL training module with
100% coverage. It uses temporary files and mocks to avoid side effects, and tests
all branches of the contract negotiation environment and RL agent behavior.

Usage:
    coverage run -m unittest discover -s peakperformance/tests

Author: Balaji Boopal
Date: March 16, 2025

Parameters:
    None

Returns:
    None
"""
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

import peakperformance.backend.train_model  # Module under test


class TestRLTrain(unittest.TestCase):
    """
    Test cases for the RL training module.
    """

    def setUp(self) -> None:
        """
        Set up dummy player data for testing.
        """
        self.dummy_df = pd.DataFrame({
            "PLAYER": ["Test Player"],
            "GROSS P/W (EUR)": [1000],
            "age": [25],
            "Rating": [7],
            "CLUB": ["Test Club"],
            "pos": ["Forward"]
        })

    def test_load_player_data_success(self) -> None:
        """
        Test that load_player_data loads CSV data correctly and drops rows with NaNs.
        """
        df = pd.DataFrame({
            "PLAYER": ["A", "B"],
            "GROSS P/W (EUR)": [1000, 2000],
            "age": [25, 30],
            "Rating": [7, 8],
            "CLUB": ["X", "Y"],
            "pos": ["Forward", "Midfielder"]
        })
        # Add an extra row with a missing value to be dropped.
        df_missing = pd.DataFrame({
            "PLAYER": ["C"],
            "GROSS P/W (EUR)": [None],
            "age": [27],
            "Rating": [6.5],
            "CLUB": ["Z"],
            "pos": ["Defender"]
        })
        df = pd.concat([df, df_missing], ignore_index=True)
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp_file:
            df.to_csv(tmp_file.name, index=False)
            tmp_file_name = tmp_file.name
        try:
            loaded_df = peakperformance.backend.train_model.load_player_data(
                tmp_file_name)
            # Expect the row with missing "GROSS P/W (EUR)" to be dropped.
            self.assertEqual(loaded_df.shape[0], 2)
            self.assertListEqual(list(loaded_df.columns), list(df.columns))
        finally:
            os.remove(tmp_file_name)

    def test_deep_q_network_forward(self) -> None:
        """
        Test that DeepQNetwork forward pass returns output of correct shape.
        """
        net = peakperformance.backend.train_model.DeepQNetwork(
            state_dim=5, action_dim=4)
        input_tensor = torch.randn(1, 5)
        output = net(input_tensor)
        self.assertEqual(output.shape, (1, 4))

    def test_contract_negotiation_environment_reset(self) -> None:
        """
        Test that ContractNegotiationEnvironment.reset returns the expected state.
        """
        env = peakperformance.backend.train_model.ContractNegotiationEnvironment(
            self.dummy_df)
        state = env.reset("Test Player")
        expected_state = np.array([25, 7, 1000, 0, 3], dtype=float)
        np.testing.assert_array_equal(state, expected_state)
        self.assertFalse(env.done)
        self.assertIsNotNone(env.player)

    def test_contract_negotiation_environment_step_reject_low_offer(self) -> None:
        """
        Test the environment step branch where the proposed wage is too low.
        """
        env = peakperformance.backend.train_model.ContractNegotiationEnvironment(
            self.dummy_df)
        env.reset("Test Player")
        # For age=25, Rating=7, current wage=1000, expected_wage = 1000 * 1.55 = 1550.
        # Low offer if proposed wage < 0.8 * 1550 = 1240.
        proposed_wage = 1200
        _, reward, counteroffer, _, done = env.step(
            action=0, proposed_wage=proposed_wage, contract_length=3
        )
        self.assertEqual(reward, -50)
        self.assertTrue(done)
        self.assertIsNotNone(counteroffer)
        self.assertIn("refused", counteroffer)

    def test_contract_negotiation_environment_step_accept_contract(self) -> None:
        """
        Test the environment step branch for action 0 when the contract is accepted.
        """
        env = peakperformance.backend.train_model.ContractNegotiationEnvironment(
            self.dummy_df)
        env.reset("Test Player")
        _, reward, _, negotiation_log, done = env.step(
            action=0, proposed_wage=1600, contract_length=4
        )
        expected_reward = 100 + (1600 / 1550) * 50
        self.assertAlmostEqual(reward, expected_reward, places=4)
        self.assertTrue(done)
        self.assertIn("Accepted", negotiation_log[0])

    def test_contract_negotiation_environment_step_accept_contract_fail(self) -> None:
        """
        Test the environment step branch for action 0 when the wage is below expectation.
        """
        env = peakperformance.backend.train_model.ContractNegotiationEnvironment(
            self.dummy_df)
        env.reset("Test Player")
        _, reward, _, _, done = env.step(
            action=0, proposed_wage=1500, contract_length=4
        )
        self.assertEqual(reward, -20)
        self.assertFalse(done)

    def test_contract_negotiation_environment_step_negotiate_high_wage_success(self) -> None:
        """
        Test the environment step branch for action 1 with a high enough proposed wage.
        """
        env = peakperformance.backend.train_model.ContractNegotiationEnvironment(
            self.dummy_df)
        env.reset("Test Player")
        _, reward, counteroffer, _, done = env.step(
            action=1, proposed_wage=1600, contract_length=4
        )
        # Calculation:
        # base_reward = (7 - 5) * 1.2 = 2.4
        # age_penalty = max(0, (25 - 30) * -0.5) = max(0, (-5)* -0.5) = 2.5
        # Total expected reward = 30 + 2.4 + 2.5 = 34.9
        expected_reward = 34.9
        self.assertAlmostEqual(reward, expected_reward, places=4)
        self.assertTrue(done)
        self.assertEqual(
            counteroffer, "Player has accepted the new wage offer!")

    def test_contract_negotiation_environment_step_negotiate_high_wage_fail(self) -> None:
        """
        Test the environment step branch for action 1 when proposed wage is too low.
        """
        env = peakperformance.backend.train_model.ContractNegotiationEnvironment(
            self.dummy_df)
        env.reset("Test Player")
        _, reward, counteroffer, negotiation_log, done = env.step(
            action=1, proposed_wage=1500, contract_length=4
        )
        base_reward = (7 - 5) * 1.2  # 2.4
        expected_reward = 10 + base_reward  # 12.4
        self.assertAlmostEqual(reward, expected_reward, places=4)
        self.assertFalse(done)
        self.assertEqual(counteroffer, "Player wants at least €1,550/week!")
        self.assertIn("1,550", negotiation_log[0])

    def test_contract_negotiation_environment_step_change_contract_length(self) -> None:
        """
        Test the environment step branch for changing the contract length.
        """
        env = peakperformance.backend.train_model.ContractNegotiationEnvironment(
            self.dummy_df)
        env.reset("Test Player")
        # Test with contract_length close to 4.
        _, reward, counteroffer, _, _ = env.step(
            action=2, proposed_wage=1600, contract_length=4
        )
        self.assertEqual(reward, 8)
        self.assertEqual(
            counteroffer, "Player prefers a 4-year contract instead!")
        # Test with contract_length far from 4.
        env.reset("Test Player")
        _, reward, counteroffer, _, _ = env.step(
            action=2, proposed_wage=1600, contract_length=6
        )
        self.assertEqual(reward, -5)
        self.assertEqual(
            counteroffer, "Player prefers a 4-year contract instead!")

    def test_contract_negotiation_environment_step_reject_offer(self) -> None:
        """
        Test the environment step branch for rejecting the offer.
        """
        env = peakperformance.backend.train_model.ContractNegotiationEnvironment(
            self.dummy_df)
        env.reset("Test Player")
        _, reward, counteroffer, negotiation_log, done = env.step(
            action=3, proposed_wage=1600, contract_length=4
        )
        self.assertEqual(reward, -10)
        self.assertTrue(done)
        self.assertIn("rejected", counteroffer)
        self.assertIn("rejected", negotiation_log[0])

    def test_reinforcement_learning_agent_select_action_exploration(self) -> None:
        """
        Test that the RL agent selects a random action during exploration.
        """
        agent = peakperformance.backend.train_model.ReinforcementLearningAgent(state_dim=5,
                                                                               action_dim=4)
        agent.epsilon = 1.0  # force exploration
        state = np.array([25, 7, 1000, 0, 3], dtype=float)
        action = agent.select_action(state)
        self.assertIn(action, [0, 1, 2, 3])

    def test_reinforcement_learning_agent_select_action_exploitation(self) -> None:
        """
        Test that the RL agent selects the best action during exploitation.
        """
        agent = peakperformance.backend.train_model.ReinforcementLearningAgent(state_dim=5,
                                                                               action_dim=4)
        agent.epsilon = 0.0  # force exploitation
        # Override the model's forward to return a fixed tensor.
        agent.model.forward = lambda x: torch.tensor([[0.1, 0.2, 0.3, 0.4]])
        state = np.array([25, 7, 1000, 0, 3], dtype=float)
        action = agent.select_action(state)
        self.assertEqual(action, 3)

    def test_reinforcement_learning_agent_save_and_load_model(self) -> None:
        """Test saving and loading the RL agent model."""
        agent = peakperformance.backend.train_model.ReinforcementLearningAgent(
            state_dim=5, action_dim=4
        )
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_path = tmp_file.name

        try:
            use_jit = "GITHUB_ACTIONS" not in os.environ
            agent.save_model(temp_path, use_jit=use_jit)
            agent.load_model(temp_path)

            # Fix isinstance issue
            self.assertIsInstance(
                agent.model, (
                    peakperformance.backend.train_model.DeepQNetwork, torch.jit.ScriptModule
                )
            )
        finally:
            os.remove(temp_path)

    @patch("peakperformance.backend.train_model.train_model")
    @patch("peakperformance.backend.train_model.load_player_data")
    def test_main(self, mock_load_player_data, mock_train_model) -> None:
        """
        Test that main loads the data and calls train_model with the loaded data.
        """
        dummy_df = pd.DataFrame({
            "PLAYER": ["Test Player"],
            "GROSS P/W (EUR)": [1000],
            "age": [25],
            "Rating": [7],
            "CLUB": ["Test Club"],
            "pos": ["Forward"]
        })
        mock_load_player_data.return_value = dummy_df
        peakperformance.backend.train_model.main()
        mock_train_model.assert_called_with(dummy_df)


if __name__ == "__main__":
    unittest.main()
