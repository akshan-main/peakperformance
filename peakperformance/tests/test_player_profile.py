"""
Unit tests for the football player profile module.

All tests are designed to pass when running:
    coverage run -m unittest discover -s peakperformance/tests

Author: Balaji Boopal
Date: March 16, 2025

Parameters:
    None

Returns:
    None
"""

import io
import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import requests
import peakperformance.pages.player_profile


class TestPlayerProfile(unittest.TestCase):
    """
    Test cases for the peakperformance.pages.player_profile module.
    """

    def setUp(self) -> None:
        """
        Set up dummy data for use in tests.
        """
        self.dummy_data = pd.DataFrame({
            "player": ["Test Player", "Test Player"],
            "season": [2021, 2022],
            "Rating": [7.5, 8.0],
            "predicted_rating": [7.0, 8.5],
            "Goals": [5, 10],
            "Assists": [3, 4],
            "Matches Played": [20, 25],
            "age": [25, 26],
            "Expected Goals": [4.5, 9.0]
        })
        self.dummy_api_data = {
            "strPlayer": "Test Player",
            "strTeam": "Test Team",
            "strThumb": "https://via.placeholder.com/150",
            "strNationality": "Testland",
            "strPosition": "Forward"
        }

    @patch("peakperformance.pages.player_profile.st.markdown")
    def test_set_theme(self, mock_markdown: MagicMock) -> None:
        """
        Test that set_theme calls st.markdown with the proper CSS style.

        Parameters:
            mock_markdown (MagicMock): The patched st.markdown function.

        Returns:
            None
        """
        peakperformance.pages.player_profile.set_theme()
        self.assertTrue(mock_markdown.called)
        args, kwargs = mock_markdown.call_args
        self.assertIn("background-color: white", args[0])
        self.assertTrue(kwargs.get("unsafe_allow_html", False))

    def test_load_player_data_success(self) -> None:
        """
        Test load_player_data returns a DataFrame when the CSV file exists.

        Parameters:
            None

        Returns:
            None
        """
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".csv") as tmp_file:
            self.dummy_data.to_csv(tmp_file.name, index=False)
            tmp_file_name = tmp_file.name

        try:
            df = peakperformance.pages.player_profile.load_player_data(
                tmp_file_name)
            self.assertFalse(df.empty)
            self.assertListEqual(
                list(df.columns), list(self.dummy_data.columns))
        finally:
            os.remove(tmp_file_name)

    @patch("peakperformance.pages.player_profile.st.error")
    def test_load_player_data_file_not_found(self, mock_error: MagicMock) -> None:
        """
        Test load_player_data returns an empty DataFrame and calls st.error
        when the file is not found.

        Parameters:
            mock_error (MagicMock): The patched st.error function.

        Returns:
            None
        """
        df = peakperformance.pages.player_profile.load_player_data(
            "non_existent_file.csv")
        self.assertTrue(df.empty)
        mock_error.assert_called_once()

    @patch("peakperformance.pages.player_profile.requests.get")
    def test_fetch_player_data_success(self, mock_get: MagicMock) -> None:
        """
        Test fetch_player_data returns player data when the API call is successful.

        Parameters:
            mock_get (MagicMock): The patched requests.get function.

        Returns:
            None
        """
        dummy_response = MagicMock()
        dummy_response.status_code = 200
        dummy_response.json.return_value = {"player": [self.dummy_api_data]}
        mock_get.return_value = dummy_response

        result = peakperformance.pages.player_profile.fetch_player_data(
            "Test Player")
        self.assertEqual(result, self.dummy_api_data)
        mock_get.assert_called_once()

    @patch("peakperformance.pages.player_profile.requests.get")
    def test_fetch_player_data_no_player(self, mock_get: MagicMock) -> None:
        """
        Test fetch_player_data returns an empty dictionary when no player data is found.

        Parameters:
            mock_get (MagicMock): The patched requests.get function.

        Returns:
            None
        """
        dummy_response = MagicMock()
        dummy_response.status_code = 200
        dummy_response.json.return_value = {"player": None}
        mock_get.return_value = dummy_response

        result = peakperformance.pages.player_profile.fetch_player_data(
            "Unknown Player")
        self.assertEqual(result, {})
        mock_get.assert_called_once()

    @patch("peakperformance.pages.player_profile.requests.get",
           side_effect=requests.RequestException("Error"))
    @patch("peakperformance.pages.player_profile.st.error")
    def test_fetch_player_data_exception(
        self, mock_error: MagicMock, _: MagicMock
    ) -> None:
        """
        Test fetch_player_data returns an empty dictionary when a RequestException is raised.

        Parameters:
            mock_error (MagicMock): The patched st.error function.
            mock_get (MagicMock): The patched requests.get function with an exception.

        Returns:
            None
        """
        result = peakperformance.pages.player_profile.fetch_player_data(
            "Test Player")
        self.assertEqual(result, {})
        mock_error.assert_called_once()

    @patch("peakperformance.pages.player_profile.st.markdown")
    def test_display_player_header(self, mock_markdown: MagicMock) -> None:
        """
        Test display_player_header calls st.markdown with expected HTML content.

        Parameters:
            mock_markdown (MagicMock): The patched st.markdown function.

        Returns:
            None
        """
        peakperformance.pages.player_profile.display_player_header(
            self.dummy_api_data)
        self.assertTrue(mock_markdown.called)
        args, _ = mock_markdown.call_args
        self.assertIn("Test Player", args[0])
        self.assertIn("Test Team", args[0])

    @patch("peakperformance.pages.player_profile.st.columns")
    def test_display_player_info(self, mock_columns: MagicMock) -> None:
        """
        Test display_player_info calls st.columns and the metric method with proper values.

        Parameters:
            mock_columns (MagicMock): The patched st.columns function.

        Returns:
            None
        """
        fake_col1 = MagicMock()
        fake_col2 = MagicMock()
        mock_columns.return_value = [fake_col1, fake_col2]

        peakperformance.pages.player_profile.display_player_info(
            self.dummy_api_data)
        fake_col1.metric.assert_called_once_with("Nationality", "Testland")
        fake_col2.metric.assert_called_once_with("Position", "Forward")

    def test_get_rating_trend_found(self) -> None:
        """
        Test get_rating_trend returns the correct DataFrame when player data is found.

        Parameters:
            None

        Returns:
            None
        """
        trend_df = peakperformance.pages.player_profile.get_rating_trend("Test Player",
                                                                         self.dummy_data)
        self.assertIsNotNone(trend_df)
        self.assertIn("season", trend_df.columns)
        self.assertIn("Rating", trend_df.columns)
        self.assertEqual(len(trend_df), 2)

    def test_get_rating_trend_not_found(self) -> None:
        """
        Test get_rating_trend returns None when no matching player data is found.

        Parameters:
            None

        Returns:
            None
        """
        trend_df = peakperformance.pages.player_profile.get_rating_trend("Nonexistent Player",
                                                                         self.dummy_data)
        self.assertIsNone(trend_df)

    def test_plot_rating_trend_found(self) -> None:
        """
        Test plot_rating_trend returns a BytesIO object when player data is found.

        Parameters:
            None

        Returns:
            None
        """
        result = peakperformance.pages.player_profile.plot_rating_trend("Test Player",
                                                                        self.dummy_data)
        self.assertIsInstance(result, io.BytesIO)
        self.assertGreater(len(result.getvalue()), 0)

    def test_plot_rating_trend_not_found(self) -> None:
        """
        Test plot_rating_trend returns None when no matching player data is found.

        Parameters:
            None

        Returns:
            None
        """
        result = peakperformance.pages.player_profile.plot_rating_trend("Nonexistent Player",
                                                                        self.dummy_data)
        self.assertIsNone(result)

    def test_get_latest_season_stats_found(self) -> None:
        """
        Test get_latest_season_stats returns correct statistics for the latest season.

        Parameters:
            None

        Returns:
            None
        """
        stats = peakperformance.pages.player_profile.get_latest_season_stats("Test Player",
                                                                             self.dummy_data)
        self.assertIsNotNone(stats)
        self.assertEqual(stats["season"], 2022)
        self.assertEqual(stats["goals"], 10)
        self.assertEqual(stats["assists"], 4)
        self.assertEqual(stats["matches"], 25)
        self.assertEqual(stats["current_rating"], 8.0)

    def test_get_latest_season_stats_not_found(self) -> None:
        """
        Test get_latest_season_stats returns None when no matching player data is found.

        Parameters:
            None

        Returns:
            None
        """
        stats = peakperformance.pages.player_profile.get_latest_season_stats("Nonexistent Player",
                                                                             self.dummy_data)
        self.assertIsNone(stats)

    @patch("peakperformance.pages.player_profile.st.selectbox", return_value="Test Player")
    def test_main_flow(self, mock_selectbox: MagicMock) -> None:
        """
        Test the main function flow ensuring the proper st calls are made.

        Parameters:
            mock_selectbox (MagicMock): The patched st.selectbox function.

        Returns:
            None
        """
        mock_player_data = {
            "strPlayer": "Test Player",
            "strTeam": "Test Team",
            "strThumb": "https://via.placeholder.com/150",
            "strNationality": "Testland",
            "strPosition": "Forward"
        }

        with patch(
            "peakperformance.pages.player_profile.fetch_player_data",
            return_value=mock_player_data
        ) as mock_fetch, \
            patch(
                "peakperformance.pages.player_profile.load_player_data",
                return_value=self.dummy_data
        ):

            with patch(
                "peakperformance.pages.player_profile.plot_rating_trend",
                return_value=io.BytesIO(b"dummy")
            ) as mock_plot_rating_trend, \
                    patch("peakperformance.pages.player_profile.get_latest_season_stats",
                          return_value={
                              "season": 2022, "goals": 10, "assists": 4,
                              "matches": 25, "current_rating": 8.0,
                          }) as mock_get_latest_season_stats:

                with patch("peakperformance.pages.player_profile.st.container") as mock_container, \
                        patch("peakperformance.pages.player_profile.st.error") as mock_error, \
                    patch(
                    "peakperformance.pages.player_profile.st.markdown"
                ) as mock_markdown, \
                        patch("peakperformance.pages.player_profile.st.image") as mock_image:

                    # Simulate container context manager behavior.
                    mock_container.return_value.__enter__.return_value = None
                    peakperformance.pages.player_profile.main()

                    mock_selectbox.assert_called_once()
                    mock_fetch.assert_called_once_with("Test Player")
                    mock_markdown.assert_called()
                    mock_image.assert_called()
                    mock_get_latest_season_stats.assert_called()
                    mock_plot_rating_trend.assert_called()
                    mock_error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
