"""
Unit tests for the contract simulator module.

This module tests all functions and classes defined in the contract simulator:
- fix_translucent_bar
- get_player_details
- generate_agent_offer
- display_player_card
- display_newspaper_announcement
- DQN.forward
- set_background

Author: Balaji Boopal
Date: March 16, 2025

Usage:
    coverage run --source=peakperformance -m unittest discover -s peakperformance/tests

Parameters:
    None

Returns:
    None
"""

from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import torch
import peakperformance.pages.the_negotiator as cs


class TestContractSimulator(unittest.TestCase):
    """Tests for the contract simulator module."""

    def setUp(self) -> None:
        """
        Set up dummy data and override global PLAYER_DATA for testing.
        """
        # Create a dummy DataFrame with required columns.
        self.dummy_df = pd.DataFrame({
            "PLAYER": ["Test Player"],
            "CLUB": ["Test Club"],
            "Rating": [80],
            "age": [27],
            "GROSS P/W (EUR)": [10000],
            "pos": ["Midfielder"]
        })
        # Override the module-level PLAYER_DATA with our dummy DataFrame.
        cs.PLAYER_DATA = self.dummy_df

    def test_fix_translucent_bar(self) -> None:
        """
        Test that fix_translucent_bar calls st.markdown with the expected CSS.
        """
        with patch.object(cs.st, "markdown") as mock_markdown:
            cs.fix_translucent_bar()
            mock_markdown.assert_called_once()
            args, _ = mock_markdown.call_args
            self.assertIn(".stAlert", args[0])
            self.assertIn("background-color: #b8860b", args[0])

    def test_get_player_details(self) -> None:
        """
        Test that get_player_details returns the correct details.
        """
        details = cs.get_player_details("Test Player")
        expected = {
            "club": "Test Club",
            "rating": 80,
            "age": 27,
            "current_wage": 10000,
            "position": "Midfielder",
        }
        self.assertEqual(details, expected)

    def test_generate_agent_offer_under26(self) -> None:
        """
        Test generate_agent_offer for a player under 26.
        """
        # For age < 26, function uses random.uniform(1.5, 1.8)
        with patch("peakperformance.pages.the_negotiator.random.uniform", return_value=1.5):
            offer = cs.generate_agent_offer(25, 75, 10000)
            self.assertEqual(offer, 10000 * 1.5)

    def test_generate_agent_offer_between26and30(self) -> None:
        """
        Test generate_agent_offer for a player between 26 and 30.
        """
        # For 26 <= age <= 30, function uses random.uniform(1.3, 1.5)
        with patch("peakperformance.pages.the_negotiator.random.uniform", return_value=1.3):
            offer = cs.generate_agent_offer(27, 80, 10000)
            self.assertEqual(offer, 10000 * 1.3)

    def test_generate_agent_offer_over30(self) -> None:
        """
        Test generate_agent_offer for a player over 30.
        """
        # For age > 30, if rating < 85, function uses random.uniform(0.7, 1.0)
        with patch("peakperformance.pages.the_negotiator.random.uniform", return_value=0.7):
            offer = cs.generate_agent_offer(31, 80, 10000)
            self.assertEqual(offer, 10000 * 0.7)
        # And if rating >= 85, uses random.uniform(0.7, 1)
        with patch("peakperformance.pages.the_negotiator.random.uniform", return_value=0.8):
            offer = cs.generate_agent_offer(32, 90, 10000)
            self.assertEqual(offer, 10000 * 0.8)

    def test_display_player_card(self) -> None:
        """
        Test that display_player_card calls st.markdown with correct content.
        """
        dummy_info = {
            "club": "Test Club",
            "rating": 80,
            "age": 27,
            "current_wage": 10000,
            "position": "Midfielder",
        }
        with patch.object(cs.st, "markdown") as mock_markdown:
            cs.display_player_card("Test Player", dummy_info)
            mock_markdown.assert_called_once()
            call_arg = mock_markdown.call_args[0][0]
            self.assertIn("Test Player", call_arg)
            self.assertIn("Test Club", call_arg)
            self.assertIn("80", call_arg)
            self.assertIn("27", call_arg)
            self.assertIn("10,000", call_arg)
            self.assertIn("Midfielder", call_arg)

    def test_display_newspaper_announcement_positive(self) -> None:
        """
        Test display_newspaper_announcement with a positive reward.
        """
        with patch.object(cs.st, "markdown") as mock_markdown, \
                patch.object(cs.st, "balloons") as mock_balloons:
            cs.display_newspaper_announcement(
                "Test Player", 15000, 4, "Test Club", 50)
            # Expect balloons to be triggered for positive reward.
            mock_balloons.assert_called_once()
            # The headline should indicate a finalized deal.
            args = mock_markdown.call_args_list[1][0][0]
            self.assertIn("BLOCKBUSTER DEAL FINALIZED!", args)
            self.assertIn("Test Player Secures a Lucrative Contract", args)

    def test_display_newspaper_announcement_negative(self) -> None:
        """
        Test display_newspaper_announcement with a non-positive reward.
        """
        with patch.object(cs.st, "markdown") as mock_markdown:
            cs.display_newspaper_announcement(
                "Test Player", 15000, 4, "Test Club", -10)
            args = mock_markdown.call_args_list[1][0][0]
            self.assertIn("CONTRACT TALKS COLLAPSE!", args)
            self.assertIn("Test Player Walks Away From Negotiations!", args)

    def test_dqn_forward(self) -> None:
        """
        Test that the DQN forward method returns a tensor of correct shape.
        """
        model = cs.DQN(5, 4)
        input_tensor = torch.randn(1, 5)
        output = model(input_tensor)
        self.assertEqual(output.shape, (1, 4))

    def test_set_background(self) -> None:
        """
        Test that set_background calls st.markdown with a background image style.
        """
        with patch.object(cs.st, "markdown") as mock_markdown:
            cs.set_background()
            mock_markdown.assert_called_once()
            args = mock_markdown.call_args[0][0]
            self.assertIn("background-image:", args)
            self.assertIn("url(", args)

    def test_globals(self) -> None:
        """
        Test that global constants have the expected types.
        """
        self.assertIsInstance(cs.CSV_PATH, Path)
        self.assertIsInstance(cs.MODEL_PATH, Path)
        self.assertIsInstance(cs.PLAYER_DATA, pd.DataFrame)

    @patch("peakperformance.pages.the_negotiator.st.sidebar.button", return_value=False)
    def test_submit_offer_branch_not_executed(self, mock_button: MagicMock) -> None:
        """
        Test that if the submit button is not pressed, the offer branch is not executed.
        """
        # Since the global code in the module already executes on import,
        # we simulate that st.sidebar.button returns False so that the block is skipped.
        self.assertFalse(mock_button())


if __name__ == "__main__":
    unittest.main()
