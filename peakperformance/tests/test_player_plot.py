"""
Unit tests for the player performance visualization in peakperformance.

This module contains tests for:
- Data loading and preprocessing functions
- Metric selection and scaling
- Radar chart generation with Plotly

These tests use unittest and unittest.mock to simulate external dependencies.

Author: Joshua Son
Date: March 17, 2025
"""

import unittest
from unittest.mock import patch
import pandas as pd
import plotly.graph_objects as go
from peakperformance.pages.player_plot import load_data, compute_scaled_values, generate_radar_chart


class TestPlayerPlot(unittest.TestCase):
    """
    Unit tests for player performance visualization in peakperformance.
    """

    @patch("peakperformance.pages.player_plot.pd.read_csv")
    def test_load_data(self, mock_read_csv):
        """Test if the dataset loads properly and renames columns."""
        mock_data = pd.DataFrame({
            "PLAYER": ["Messi", "Ronaldo"],
            "Season": [2017, 2023],
            "G": [30, 25],
            "A": [10, 12],
            "per 90": [1.2, 1.5]
        })
        mock_read_csv.return_value = mock_data

        for filepath in [None, "dummy/path/to/data.csv"]:
            df, metrics = load_data(filepath)

            self.assertIn("G", df.columns)
            self.assertIn("A", df.columns)
            self.assertIn("per 90", df.columns)
            self.assertNotIn("PLAYER", metrics)
            self.assertNotIn("Season", metrics)

            if filepath:
                mock_read_csv.assert_called_with(filepath)

    def test_compute_scaled_values(self):
        """Test the compute_scaled_values function."""
        df = pd.DataFrame({
            "PLAYER": ["Messi"],
            "Season": [2017],
            "G": [30],
            "A": [10],
            "per 90": [1.2]
        })
        selected_metrics = ["G", "A", "per 90"]
        player_row = df.iloc[0]

        result = compute_scaled_values(df, selected_metrics, player_row)
        expected_scaled_values = [100.0, 100.0, 100.0, 100.0]

        self.assertEqual(result["scaled_values"], expected_scaled_values)

    def test_generate_radar_chart(self):
        """Test if radar chart generation runs successfully."""
        df = pd.DataFrame({
            "PLAYER": ["Messi"],
            "Season": [2017],
            "G": [30],
            "A": [10],
            "per 90": [1.2]
        })
        selected_metrics = ["G", "A", "per 90"]
        fig = generate_radar_chart(df, df, selected_metrics)

        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 1, "Radar chart should have one trace per player.")

if __name__ == "__main__":
    unittest.main()
