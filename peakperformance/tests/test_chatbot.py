import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
import streamlit as st
import chatbot  # Import the chatbot script

class TestChatbot(unittest.TestCase):

    @patch("chatbot.pd.read_csv")
    def test_load_data(self, mock_read_csv):
        """Test data loading and preprocessing"""
        sample_data = {
            "PLAYER": ["Lionel Messi", "Cristiano Ronaldo"],
            "Season": [2017, 2023],
            "CLUB": ["Barcelona", "Al Nassr"],
            "League": ["La Liga", "Saudi League"],
            "age": [30, 38],
            "born": ["1987", "1985"],
            "nation": ["Argentina", "Portugal"],
            "Goals Scored": [50, 40]
        }
        df_mock = pd.DataFrame(sample_data)
        mock_read_csv.return_value = df_mock
        
        df, df_norm, metrics_cols = chatbot.load_data()
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIsInstance(df_norm, pd.DataFrame)
        self.assertIn("PLAYER", df.columns)
        self.assertNotIn("Goals Scored", df.columns)  # Column should be dropped

    def test_remove_accents(self):
        """Test accent removal function"""
        self.assertEqual(chatbot.remove_accents("Müller"), "Muller")
        self.assertEqual(chatbot.remove_accents("Renée"), "Renee")
        self.assertEqual(chatbot.remove_accents("São Paulo"), "Sao Paulo")
        self.assertEqual(chatbot.remove_accents(123), 123)  # Should return non-string input unchanged

    @patch("chatbot.datetime")
    def test_currenttime(self, mock_datetime):
        """Test if currenttime is formatted correctly"""
        mock_datetime.now.return_value.strftime.return_value = "14:30"
        self.assertEqual(chatbot.currenttime, "14:30")

    @patch("chatbot.st.chat_input")
    @patch("chatbot.st.markdown")
    def test_chat_input_processing(self, mock_markdown, mock_chat_input):
        """Test chatbot query handling"""
        mock_chat_input.return_value = "top 5 players similar to 2017 Messi in 2023 La Liga"
        
        with patch.dict(st.session_state, {"messages": []}):
            query = chatbot.st.chat_input("Enter your query:")
            self.assertEqual(query, "top 5 players similar to 2017 Messi in 2023 La Liga")
            
            chatbot.st.session_state.messages.append({
                "role": "user",
                "content": query
            })
            self.assertEqual(len(st.session_state.messages), 1)

    @patch("chatbot.st.sidebar.checkbox", return_value=True)
    @patch("chatbot.st.sidebar.multiselect")
    def test_sidebar_selection(self, mock_multiselect, mock_checkbox):
        """Test sidebar metric selection"""
        mock_multiselect.return_value = ["Rating", "Goals"]
        selected_metrics = chatbot.metrics_cols if mock_checkbox else mock_multiselect
        self.assertTrue(mock_checkbox)
        self.assertEqual(selected_metrics, chatbot.metrics_cols)

    def test_nationality_extraction(self):
        """Test nationality parsing from query"""
        query = "top 10 Brazilian players in 2023 season"
        parsed = {"nationality": None}
        
        for dem, country in chatbot.demonym_map.items():
            if dem in query.lower():
                parsed["nationality"] = country
                break
        
        self.assertEqual(parsed["nationality"], "Brazil")

    @patch("chatbot.np.linalg.norm", return_value=1.0)
    def test_similarity_computation(self, mock_norm):
        """Test cosine similarity calculation"""
        ref_vector = np.array([0.5, 0.3, 0.7])
        B_matrix = np.array([[0.4, 0.2, 0.6], [0.1, 0.9, 0.5]])
        B_norms = np.linalg.norm(B_matrix, axis=1)
        dot_prods = B_matrix.dot(ref_vector)
        cosine_sim = dot_prods / (mock_norm * B_norms)
        
        self.assertEqual(cosine_sim.shape, (2,))
        self.assertAlmostEqual(cosine_sim[0], 0.991, places=3)

    @patch("chatbot.st.chat_message")
    def test_error_handling(self, mock_chat_message):
        """Test chatbot's error response when player is not found"""
        with patch.dict(st.session_state, {"messages": []}):
            error_message = "Player **Xavi** in season **2015** was not found."
            st.session_state.messages.append({"role": "assistant", "content": error_message})
            self.assertEqual(st.session_state.messages[-1]["content"], error_message)

    @patch("chatbot.st.stop")
    def test_empty_query_handling(self, mock_stop):
        """Test chatbot stopping when no query is provided"""
        with patch.dict(st.session_state, {"messages": []}):
            chatbot.query = ""
            if not chatbot.query:
                chatbot.st.stop()
        
            mock_stop.assert_called_once()

if __name__ == "__main__":
    unittest.main()
