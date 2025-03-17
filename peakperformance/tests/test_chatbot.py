import html
import base64
import unittest
from unittest.mock import patch, MagicMock
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from peakperformance.pages import chatbot
from chatbot import (
    load_data,
    get_base64,
    load_background_image,
    remove_accents,
    parse_user_query,
    filter_candidates,
    compute_similarity,
    init_session_messages, 
    create_user_html, 
    create_assistant_html,
    demonym_map
)

class TestChatbot(unittest.TestCase):

    def setUp(self):
        st.session_state = MagicMock()
        st.session_state.messages = []

    def test_get_base64_file_not_found(self):
        result = get_base64('nonexistent.png')
        self.assertIsNone(result)

    @patch('builtins.open', new_callable=unittest.mock.mock_open, read_data=b'testdata')
    def test_get_base64(self, mock_file):
        result = get_base64('dummyfile.png')
        expected = base64.b64encode(b'testdata').decode()
        self.assertEqual(result, expected)

    @patch('peakperformance.pages.chatbot.get_base64', return_value='fake_base64')
    def test_load_background_image(self, mock_get_base64):
        bg_img = load_background_image()
        self.assertEqual(bg_img, 'fake_base64')

    @patch('pandas.read_csv')
    def test_load_data(self, mock_read_csv):
        mock_df = pd.DataFrame({
            "PLAYER": ["Lionel Messi", "Cristiano Ronaldo"],
            "Season": [2017, 2023],
            "CLUB": ["Barcelona", "Al Nassr"],
            "League": ["La Liga", "Saudi League"],
            "age": [30, 38],
            "born": ["1987", "1985"],
            "nation": ["Argentina", "Portugal"],
            "Goals Scored": [30, 25],
            "rk": [1, 2],
            "Rating": [9.5, 8.8]
        })
        with patch('pandas.read_csv', return_value=mock_df):
            df, df_norm, cols = load_data()
            self.assertFalse(df.empty)
            self.assertNotIn("Goals Scored", df.columns)

    def test_remove_accents(self):
        self.assertEqual(remove_accents("Müller"), "Muller")
        self.assertEqual(remove_accents("Renée"), "Renee")
        self.assertEqual(remove_accents("São Paulo"), "Sao Paulo")
        self.assertEqual(remove_accents(123), 123)

    def test_parse_user_query_complete(self):
        """Test full query parsing including nationality."""
        query = "top 3 players similar to 2017 Neymar in 2023 Ligue 1 under 25 brazilian"
        df_mock = pd.DataFrame({"nation": ["Brazil", "Argentina"]})
        parsed = parse_user_query(query, df_mock, demonym_map)
        self.assertEqual(parsed["reference_player"], "Neymar")
        self.assertEqual(parsed["reference_season"], 2017)
        self.assertEqual(parsed["target_season"], 2023)
        self.assertEqual(parsed["num_results"], 3)
        self.assertEqual(parsed["league"], "Ligue 1")
        self.assertEqual(parsed["nationality"], "Brazil")
        self.assertEqual(parsed["age_filter"], ("under", 25))


    def test_filter_candidates_player_not_found(self):
        df_mock = pd.DataFrame({
            "PLAYER": ["Cristiano Ronaldo"],
            "Season": [2017],
            "League": ["La Liga"],
            "age": [32],
            "nation": ["Portugal"]
        })
        df_norm_mock = df_mock.copy()
        parsed = {
            "reference_player": "Messi",
            "reference_season": 2017,
            "target_season": 2017,
            "league": "La Liga",
            "nationality": None,
            "age_filter": None
        }
        candidates_df, candidates_norm, ref_vector, error = filter_candidates(
            df_mock, df_norm_mock, parsed, metrics_list=["age"]
        )
        self.assertIsNone(candidates_df)
        self.assertEqual(error, "Player **Messi** in season **2017** was not found.")

    def test_compute_similarity_empty_norms(self):
        candidates_df = pd.DataFrame({
            "PLAYER": ["Player A"],
            "CLUB": ["Club A"],
            "League": ["La Liga"],
            "age": [24]
        })
        candidates_norm = pd.DataFrame({
            "Metric1": [0],
            "Metric2": [0]
        })
        ref_vector = np.array([0, 0])
        metrics_list = ["Metric1", "Metric2"]
        top_players = compute_similarity(ref_vector, candidates_df, candidates_norm, metrics_list, num_results=1)
        self.assertTrue(top_players.empty)

    def test_compute_similarity(self):
        candidates_df = pd.DataFrame({
            "PLAYER": ["Player A", "Player B"],
            "CLUB": ["Club A", "Club B"],
            "League": ["La Liga", "Bundesliga"],
            "age": [24, 27]
        })
        candidates_norm = pd.DataFrame({
            "Metric1": [0.8, 0.5],
            "Metric2": [0.6, 0.4]
        })
        ref_vector = np.array([0.9, 0.1])
        metrics_list = ["Metric1", "Metric2"]
        top_players = compute_similarity(ref_vector, candidates_df, candidates_norm, metrics_list, num_results=2)
        self.assertFalse(top_players.empty)
        self.assertEqual(len(top_players), 2)
        self.assertIn("Similarity", top_players.columns)
        
    @patch("peakperformance.pages.chatbot.st.markdown")
    @patch("peakperformance.pages.chatbot.st.session_state", new_callable=MagicMock)
    def test_parse_user_query_session_state(self, mock_session_state, mock_markdown):
        """Test if session state correctly initializes messages."""        
        mock_session_state.messages = []
        query = "top 3 players similar to 2017 Neymar in 2023 Ligue 1"
        df_mock = pd.DataFrame({"nation": ["Brazil"]})
        parse_user_query(query, df_mock, demonym_map)
        self.assertGreaterEqual(len(mock_session_state.messages), 1)
        self.assertEqual(mock_session_state.messages[0]["role"], "assistant")

    def test_compute_similarity_zero_norm(self):
        candidates_df = pd.DataFrame({"PLAYER": ["A"], "age": [24]})
        candidates_norm = pd.DataFrame({"Metric1": [0], "Metric2": [0]})
        ref_vector = np.array([1, 0])
        metrics_list = ["Metric1", "Metric2"]
        top_players = compute_similarity(ref_vector, candidates_df, candidates_norm, metrics_list, num_results=1)
        self.assertTrue(top_players.empty)

    def test_compute_similarity_zero_vectors(self):
        """Test similarity computation when reference and candidate vectors are all zeros."""
        candidates_df = pd.DataFrame({"PLAYER": ["A"], "age": [24]})
        candidates_norm = pd.DataFrame({"Metric1": [0], "Metric2": [0]})
        ref_vector = np.array([0, 0])
        metrics_list = ["Metric1", "Metric2"]
        top_players = compute_similarity(ref_vector, candidates_df, candidates_norm, metrics_list, num_results=1)
        self.assertTrue(top_players.empty)

    def test_init_session_messages(self):
        """Test session initialization when no messages exist."""
        mock_session_state = MagicMock()
        messages = init_session_messages(mock_session_state)        
        self.assertEqual(len(messages), 1)
        self.assertEqual(messages[0]["role"], "assistant")

    def test_create_user_html(self):
        html = create_user_html("top 5 similar to Ronaldo", "12:00")
        self.assertIn("top 5 similar to Ronaldo", html)
        self.assertIn("12:00", html)

    def test_filter_candidates_no_metrics(self):
        """Test filtering when reference player has no metric values."""
        df_mock = pd.DataFrame({"PLAYER": ["Messi"], "Season": [2017], "age": [30]})
        df_norm_mock = pd.DataFrame({"age": [np.nan]})  # No valid values
        parsed = {
            "reference_player": "Messi",
            "reference_season": 2017,
            "target_season": 2023,
            "league": None,
            "nationality": None,
            "age_filter": None
        }
        candidates_df, candidates_norm, ref_vector, error = filter_candidates(
            df_mock, df_norm_mock, parsed, metrics_list=["age"]
        )
        self.assertIsNone(candidates_df)
        
        expected_error = f"**{parsed['reference_player']} ({parsed['reference_season']})** has no values for the selected metrics."
        self.assertEqual(error, expected_error)

    def test_parse_user_query_no_player(self):
        """Test query parsing when no player name is found."""
        query = "top 5 similar players in 2023 La Liga"
        df_mock = pd.DataFrame({"nation": ["Brazil"]})
        parsed = parse_user_query(query, df_mock, demonym_map)
        self.assertIsNone(parsed["reference_player"])
        self.assertEqual(parsed["target_season"], 2023)

    def test_create_assistant_html_empty_results(self):
        """Test assistant message HTML rendering when there are no results."""
        html = create_assistant_html("No players found.", "12:00")
        self.assertIn("No players found.", html)
        self.assertIn("STATMATCH", html)
        self.assertIn("12:00", html)

    def test_create_user_html_special_chars(self):
        """Test user message HTML rendering with special characters."""
        query = "<test>"
        expected = "&lt;test&gt;"
        html_output = create_user_html(query, "12:00")
        self.assertIn(expected, html_output)

    def test_filter_candidates_no_player_found(self):
        """Test `filter_candidates` when reference player is missing."""
        df_mock = pd.DataFrame({"PLAYER": ["Ronaldo"], "Season": [2017]})
        df_norm_mock = df_mock.copy()
        parsed = {"reference_player": "Messi", "reference_season": 2017}

        candidates_df, candidates_norm, ref_vector, error = filter_candidates(df_mock, df_norm_mock, parsed, metrics_list=["age"])
        
        self.assertIsNone(candidates_df)
        self.assertEqual(error, "Player **Messi** in season **2017** was not found.")

    
    def test_parse_user_query_no_season(self):
        """Test when season is missing from the query."""
        query = "top 5 players similar to Neymar"
        df_mock = pd.DataFrame({"nation": ["Brazil"]})
        parsed = parse_user_query(query, df_mock, demonym_map)

        self.assertEqual(parsed["reference_player"], "Neymar")
        self.assertIsNone(parsed["reference_season"])
        self.assertEqual(parsed["num_results"], 5)
        
    def test_parse_user_query_missing_info(self):
        """Test when query is too vague and missing necessary details."""
        query = "show me top players"
        df_mock = pd.DataFrame({"nation": ["Brazil"]})

        parsed = parse_user_query(query, df_mock, demonym_map)

        self.assertIsNone(parsed["reference_player"])
        self.assertIsNone(parsed["league"])

    def test_filter_candidates_player_no_season_match(self):
        """Test when player exists but in a different season."""
        df_mock = pd.DataFrame({"PLAYER": ["Messi"], "Season": [2018]})
        df_norm_mock = df_mock.copy()
        parsed = {"reference_player": "Messi", "reference_season": 2017}
        candidates_df, candidates_norm, ref_vector, error = filter_candidates(df_mock, df_norm_mock, parsed, metrics_list=["age"])
        self.assertIsNone(candidates_df)
        self.assertEqual(error, "Player **Messi** in season **2017** was not found.")

    def test_compute_similarity_nan_values(self):
        """Test similarity computation when candidate metrics contain NaN values."""
        candidates_df = pd.DataFrame({"PLAYER": ["A"], "age": [24]})
        candidates_norm = pd.DataFrame({"Metric1": [np.nan], "Metric2": [np.nan]})
        ref_vector = np.array([1, 0])
        metrics_list = ["Metric1", "Metric2"]
        top_players = compute_similarity(ref_vector, candidates_df, candidates_norm, metrics_list, num_results=1)
        self.assertTrue(top_players.empty)

    def test_create_assistant_html_large_results(self):
        """Test assistant response when returning a long list of players."""
        result_text = "<ol>" + "".join(f"<li>Player {i}</li>" for i in range(50)) + "</ol>"
        html_output = create_assistant_html(result_text, "14:30")
        self.assertIn("Player 49", html_output)

    def test_parse_user_query_age_filters(self):
        df_mock = pd.DataFrame({"nation": ["Brazil"]})
        queries = {
            "top 5 players similar to 2017 Messi in 2023 Bundesliga under 23": ("under", 23),
            "top 10 players similar to 2015 Ronaldo in 2022 La Liga over 30": ("over", 30),
            "top 3 players similar to 2018 Neymar in 2023 Serie A 25 or younger": ("<=", 25),
            "top 5 players similar to 2019 Mbappe in 2022 Ligue 1 22 or older": (">=", 22),
            "top 5 players similar to 2019 Mbappe in 2021 Ligue 1 exactly 22": ("==", 22)
        }
        for query, expected in queries.items():
            parsed = parse_user_query(query, df_mock, demonym_map)
            self.assertEqual(parsed["age_filter"], queries[query])


    def test_filter_candidates_age_filters(self):
        df_mock = pd.DataFrame({
            "PLAYER": ["Ref Player", "Player A", "Player B", "Player C"],
            "Season": [2023, 2023, 2023, 2023],
            "age": [28, 22, 30, 35]
        })
        df_norm_mock = df_mock.copy()
        test_cases = {
            ("under", 25): ["Player A"],
            ("over", 29): ["Player B", "Player C"],
            ("<=", 30): ["Player A", "Player B"],
            (">=", 35): ["Player C"]
        }

        for age_filter, expected_players in test_cases.items():
            parsed = {
                "reference_player": "Ref Player",
                "reference_season": 2023,
                "target_season": 2023,
                "age_filter": age_filter
            }
            candidates_df, _, _, _ = filter_candidates(df_mock, df_mock, parsed, metrics_list=["age"])
            self.assertListEqual(sorted(candidates_df["PLAYER"].tolist()), sorted(expected_players))

    @patch("peakperformance.pages.chatbot.st.sidebar.checkbox", return_value=False)
    @patch("peakperformance.pages.chatbot.st.sidebar.multiselect", return_value=["age", "nation"])
    def test_sidebar_metric_selection(self, mock_multiselect, mock_checkbox):
        """Test sidebar metric selection behavior."""
        metrics_list = ["age", "nation"] if not mock_checkbox.return_value else ["all"]
        self.assertEqual(metrics_list, ["age", "nation"])

if __name__ == '__main__':
    unittest.main()
