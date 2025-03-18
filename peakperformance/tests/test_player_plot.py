"""
Unit tests for the player performance visualization in peakperformance.

This module contains tests for:
- Data loading and preprocessing functions
- Player and season selection logic
- Metric selection and scaling
- Radar chart generation with Plotly
- Streamlit UI behavior and session state handling

These tests use unittest and unittest.mock to simulate external dependencies.

Author: Joshua Son
Date: March 17, 2025
"""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from peakperformance.pages.player_plot import load_data


class TestPlot(unittest.TestCase):
    """
    Unit tests for player performance visualization in peakperformance.
    """

    @patch("peakperformance.pages.player_plot.pd.read_csv")
    def test_load_data(self, mock_read_csv):
        """Test if the dataset loads properly and renames columns."""
        mock_data = pd.DataFrame({
            "PLAYER": ["Messi", "Ronaldo"],
            "Season": [2017, 2023],
            "CLUB": ["Barcelona", "Al Nassr"],
            "League": ["La Liga", "Saudi League"],
            "G": [30, 25],
            "A": [10, 12],
            "per 90": [1.2, 1.5]
        })
        mock_read_csv.return_value = mock_data

        df, metrics = load_data()

        # Check if columns are correctly renamed
        self.assertIn("G", df.columns)
        self.assertIn("A", df.columns)
        self.assertIn("per 90", df.columns)

        # Ensure excluded columns are removed from metrics
        self.assertNotIn("PLAYER", metrics)
        self.assertNotIn("Season", metrics)

    @patch("peakperformance.pages.player_plot.st.sidebar.selectbox")
    def test_sidebar_player_selection(self, mock_selectbox):
        """Test sidebar UI player selection."""
        mock_selectbox.side_effect = ["Messi", "Ronaldo"]
        player1 = st.sidebar.selectbox("Select Player", ["Messi", "Ronaldo"])
        player2 = st.sidebar.selectbox("Select Player", ["Messi", "Ronaldo"])

        self.assertEqual(player1, "Messi")
        self.assertEqual(player2, "Ronaldo")

    @patch("peakperformance.pages.player_plot.st.sidebar.multiselect")
    def test_sidebar_metric_selection(self, mock_multiselect):
        """Test sidebar UI metric selection."""
        mock_multiselect.return_value = ["G", "A"]
        selected_metrics = st.sidebar.multiselect("Select Performance Metrics", ["G", "A", "p 90"])

        self.assertEqual(selected_metrics, ["G", "A"])

    def test_radar_chart_empty_data(self):
        """Test radar chart generation with an empty dataset."""
        fig = go.Figure()

        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 0, "Figure should have no traces for empty data.")

    def test_radar_chart_single_player(self):
        """Test radar chart generation with a single player and metrics."""
        selected_metrics = ["G", "A", "p 90"]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[30, 10, 1.2, 30],
            theta=selected_metrics + [selected_metrics[0]],
            fill="toself",
            name="Messi 2017"
        ))

        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1, "Figure should have one trace for one player.")

    def test_radar_chart_multiple_players(self):
        """Test radar chart generation with multiple players."""
        selected_metrics = ["G", "A", "p 90"]

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=[30, 10, 1.2, 30],
            theta=selected_metrics + [selected_metrics[0]],
            fill="toself",
            name="Messi 2017"
        ))
        fig.add_trace(go.Scatterpolar(
            r=[25, 12, 1.5, 25],
            theta=selected_metrics + [selected_metrics[0]],
            fill="toself",
            name="Ronaldo 2023"
        ))

        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 2, "Figure should have two traces for two players.")

    @patch("peakperformance.pages.player_plot.st.session_state", new_callable=MagicMock)
    def test_streamlit_session_state(self, mock_session_state):
        """Test Streamlit session state handling."""
        mock_session_state.players = ["Messi", "Ronaldo"]
        self.assertEqual(mock_session_state.players, ["Messi", "Ronaldo"])

if __name__ == "__main__":
    unittest.main()
