import sys
import os
import unittest
from unittest.mock import patch, mock_open, MagicMock
import base64

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from peakperformance import home

class TestHome(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        """Set up reusable test variables."""
        cls.mock_b64 = base64.b64encode(b'test data').decode()

    @patch("builtins.open", new_callable=mock_open, read_data=b'test data')
    def test_get_base64(self, mock_file):
        """Test if get_base64 reads file and encodes it correctly."""
        result = home.get_base64("dummy.jpeg")
        self.assertEqual(result, self.mock_b64)

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=b'test data')
    @patch("home.get_base64")
    @patch("streamlit.markdown")
    def test_set_background(self, mock_markdown, mock_get_base64, mock_file, mock_exists):
        """Test if set_background correctly injects base64-encoded image."""
        mock_get_base64.return_value = base64.b64encode(b'test data').decode()

        home.set_background("background.jpeg")
        mock_markdown.assert_called_once()
        args, _ = mock_markdown.call_args
        self.assertIn(mock_get_base64.return_value, args[0])

    @patch("streamlit.markdown")
    def test_inject_custom_styles(self, mock_markdown):
        """Test if inject_custom_styles applies the correct CSS."""
        home.inject_custom_styles()
        mock_markdown.assert_called_once()
        args, _ = mock_markdown.call_args
        self.assertIn(".image-button:hover", args[0])

    def test_get_abs_path(self):
        """Test absolute path generation."""
        relative_path = "test_image.jpg"
        abs_path = home.get_abs_path(relative_path)
        expected_path = os.path.abspath(os.path.join(os.path.dirname(home.__file__), "..", "assets", relative_path))
        self.assertEqual(abs_path, expected_path)

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=b'test data')
    @patch("streamlit.markdown")
    def test_display_image_button(self, mock_markdown, mock_file, mock_exists):
        """Test if display_image_button correctly formats HTML."""
        home.display_image_button("test.png", "large")
        mock_markdown.assert_called_once()
        args, _ = mock_markdown.call_args
        self.assertIn("large-button", args[0])
        self.assertIn(self.mock_b64, args[0])  # Check if encoded image is present

    @patch("os.path.exists", return_value=False)
    @patch("streamlit.error")
    def test_display_image_button_file_not_found(self, mock_error, mock_exists):
        """Test display_image_button when image file is missing."""
        home.display_image_button("missing.png", "large")
        mock_error.assert_called_once()
        args, _ = mock_error.call_args
        self.assertIn("Image not found", args[0])

    @patch("os.path.exists", return_value=True)
    @patch("builtins.open", new_callable=mock_open, read_data=b'test data')
    @patch("home.display_image_button")
    @patch("streamlit.button", return_value=True)
    @patch("streamlit.switch_page")
    @patch("streamlit.markdown")  # Mocking markdown to prevent Streamlit errors
    def test_fifa_button_file_exists_with_target(
        self, mock_markdown, mock_switch_page, mock_button, mock_display_img, mock_file, mock_exists
    ):
        """Test fifa_button when the image file exists and a target page is given."""
        home.fifa_button("exists.png", "large", "target.py")

        mock_switch_page.assert_called_once_with("target.py")

    @patch("os.path.exists", return_value=False)
    @patch("streamlit.error")
    def test_fifa_button_file_not_exists(self, mock_error, mock_exists):
        """Test fifa_button when image file is missing."""
        home.fifa_button("missing.png", "large", "target.py")

        mock_error.assert_called_once()
        args, _ = mock_error.call_args
        self.assertIn("Image not found", args[0])

    @patch("peakperformance.home.set_background")
    @patch("peakperformance.home.inject_custom_styles")
    @patch("peakperformance.home.fifa_button")
    def test_main(self, mock_fifa_button, mock_styles, mock_bg):
        """Test if main() executes without errors and calls background setup."""
        home.main()
        mock_bg.assert_called_once_with(home.get_abs_path('background.jpeg'))
        mock_styles.assert_called_once()
        self.assertEqual(mock_fifa_button.call_count, 4)


    @patch("os.path.exists", return_value=False)
    @patch("streamlit.error")
    def test_set_background_file_missing(self, mock_error, mock_exists):
        """Test set_background when file is missing."""
        home.set_background("nonexistent.jpeg")
        mock_error.assert_called_once()



if __name__ == "__main__":
    unittest.main()
