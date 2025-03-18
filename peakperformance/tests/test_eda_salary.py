"""
Unit tests for the salary module in the peakperformance application.

This module contains tests for:
- Data loading and filtering functions
- Scatter and bar chart generation
- Machine learning model training
- Salary prediction logic

These tests use unittest and unittest.mock to simulate external dependencies.

Author: Akshan Krithick
Date: March 17, 2025
"""
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from peakperformance.pages.eda_salary import (
    load_data,
    filter_data,
    plot_scatter_chart,
    plot_bar_chart,
    train_model,
    predict_salary
)

class TestSalaryModule(unittest.TestCase):
    """
    Unit tests for the salary-related functions in the peakperformance application.

    This class tests:
    - Data processing and filtering
    - Visualization functions (scatter and bar charts)
    - Machine learning model training and salary predictions
    """
    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        """Test if `load_data` correctly loads and processes salary data from a CSV file."""
        mock_df = pd.DataFrame({
            'Rating': [7.5, 8.0],
            'GROSS P/Y (EUR)': [1000000, 2000000],
            'League': ['La Liga', 'Premier League'],
            'Season': ['2022', '2023'],
            'pos': ['FW', 'MF'],
            'CLUB': ['Barcelona', 'Manchester United']
        })
        mock_read_csv.return_value = mock_df

        df = load_data()
        self.assertEqual(len(df), 2)
        self.assertNotIn('COUNTRY', df.columns)
        self.assertNotIn('POS.', df.columns)

    def test_filter_data(self):
        """
        Test if `filter_data` correctly filters player salary 
        data based on user-selected criteria.
        """
        df = pd.DataFrame({
            'Season': ['2022', '2023'],
            'League': ['La Liga', 'Premier League'],
            'pos': ['FW', 'MF'],
            'CLUB': ['Barcelona', 'Manchester United']
        })

        filtered = filter_data(df, '2022', ['La Liga'], ['FW'], ['Barcelona'])
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered.iloc[0]['CLUB'], 'Barcelona')

        filtered_all = filter_data(df, 'All', [], [], [])
        self.assertEqual(len(filtered_all), 2)

    def test_plot_scatter_chart(self):
        """Test if `plot_scatter_chart` correctly generates a scatter plot visualization."""
        df = pd.DataFrame({
            'PLAYER': ['Messi', 'Ronaldo'],
            'Rating': [8.5, 7.8],
            'GROSS P/Y (EUR)': [5000000, 4000000],
            'League': ['La Liga', 'Serie A'],
            'Season': ['2022', '2023'],
            'pos': ['FW', 'FW'],
            'CLUB': ['Barcelona', 'Juventus']
        })

        chart = plot_scatter_chart(df, 'League')
        self.assertIsNotNone(chart)
        self.assertEqual(chart.to_dict()["mark"]["type"], "circle")

    def test_plot_bar_chart(self):
        """Test if `plot_bar_chart` correctly generates a bar chart visualization."""
        df = pd.DataFrame({
            'League': ['La Liga', 'La Liga', 'Serie A'],
            'GROSS P/Y (EUR)': [5000000, 6000000, 4000000],
            'CLUB': ['Barcelona', 'Real Madrid', 'Juventus']
        })

        chart = plot_bar_chart(df)
        self.assertIsNotNone(chart)
        self.assertEqual(chart.to_dict()["mark"]["type"], "bar")

    @patch('peakperformance.pages.eda_salary.train_test_split')
    def test_train_model(self, mock_split):
        """Test if `train_model` correctly trains a machine learning model on salary data."""
        df = pd.DataFrame({
            'League': ['La Liga', 'Premier League'],
            'pos': ['FW', 'MF'],
            'Rating': [7.5, 8.0],
            'age': [25, 27],
            'GROSS P/Y (EUR)': [1000000, 2000000]
        })

        mock_split.return_value = (df[['League', 'pos', 'Rating', 'age']],
                                   df[['League', 'pos', 'Rating', 'age']],
                                   np.log(df['GROSS P/Y (EUR)']),
                                   np.log(df['GROSS P/Y (EUR)']))

        model, max_rating = train_model(df)
        self.assertIsNotNone(model)
        self.assertEqual(max_rating, {'La Liga': 7.5, 'Premier League': 8.0})

    def test_predict_salary(self):
        """Test salary prediction with a mocked model using the updated function signature."""
        model_mock = MagicMock()
        expected_salary = 2_000_000  # 2 million EUR
        model_mock.predict.return_value = [np.log(expected_salary)]
        player_info = {
            "League": "La Liga",
            "pos": "FW",
            "Rating": 10.0,
            "age": 25
        }
        max_rating_by_league = {"La Liga": 9.5}
        predicted_salary = predict_salary(model_mock, player_info, max_rating_by_league)
        self.assertAlmostEqual(predicted_salary, expected_salary, places=0)
        model_mock.predict.assert_called_once()


if __name__ == '__main__':
    unittest.main()
