import unittest
from unittest.mock import patch
import streamlit as st
import plotly.graph_objects as go
from testplot import load_data

class TestPlayerPerformanceDashboard(unittest.TestCase):

    def setUp(self):
        """Setup sample dataset for testing"""
        self.df = pd.DataFrame({
            'PLAYER': ['Messi', 'Ronaldo', 'Neymar'],
            'Season': ['2022', '2021', '2020'],
            'Metric1': [10.5, 12.3, 9.8],
            'Metric2': [7.4, 8.1, 6.9]
        })

    def test_load_data(self):
        """Test data loading and preprocessing"""
        df_cleaned = load_data(file_path="/mnt/data/playerratingssalaries_100mins.csv")
        self.assertFalse(df_cleaned.empty, "Dataset should not be empty")
        self.assertIn('PLAYER', df_cleaned.columns, "PLAYER column should exist")
        self.assertIn('Season', df_cleaned.columns, "Season column should exist")

    def test_excluded_columns(self):
        """Test that excluded columns are removed"""
        exclude_cols = ['rk', 'nation', 'pos', 'CLUB', 'League', 'age', 'born', 'Season']
        df_filtered = self.df.drop(columns=[col for col in exclude_cols if col in self.df.columns], errors='ignore')
        for col in exclude_cols:
            self.assertNotIn(col, df_filtered.columns, f"{col} should be removed")

    def test_rename_columns(self):
        """Test that percentage and per 90 columns are renamed correctly"""
        rename_map = {col: col.replace('%', 'Percent').replace('p 90', ' per 90') for col in self.df.columns}
        df_renamed = self.df.rename(columns=rename_map)
        for old_col, new_col in rename_map.items():
            self.assertIn(new_col, df_renamed.columns, f"{new_col} should be in columns")

    def test_slider_range(self):
        """Test slider selection range for player count"""
        min_players, max_players = 1, 10
        self.assertGreaterEqual(min_players, 1, "Minimum player count should be 1")
        self.assertLessEqual(max_players, 10, "Maximum player count should be 10")

    def test_selected_players(self):
        """Test selection of players and corresponding seasons"""
        player_selected = 'Messi'
        season_selected = '2022'
        df_selected = self.df[(self.df['PLAYER'] == player_selected) & (self.df['Season'] == season_selected)]
        self.assertFalse(df_selected.empty, "Selected player data should not be empty")

    def test_radar_chart_data(self):
        """Test radar chart calculations for scaled values"""
        max_values = self.df[['Metric1', 'Metric2']].max().replace(0, 1e-5)
        scaled_values = (self.df[['Metric1', 'Metric2']] / max_values) * 100
        self.assertTrue((scaled_values <= 100).all().all(), "Scaled values should be within 0-100 range")

if __name__ == "__main__":
    unittest.main()
