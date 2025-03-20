"""
FIFA 15 Style Homepage for the PeakPerformance Application.

This Streamlit-based homepage provides an interactive UI where users can:
- Set a FIFA-style background
- Inject custom UI styles for buttons and images
- Display and interact with image buttons that navigate to different app pages

Functions:
- `get_base64(bin_file)`: Encodes a file in Base64 format.
- `get_abs_path(relative_path)`: Resolves the absolute path of assets.
- `set_background(image_path)`: Sets the background image using Base64 encoding.
- `inject_custom_styles()`: Injects custom CSS styles into the Streamlit app.
- `display_image_button(image_file, size)`: Displays an image-based button.
- `fifa_button(image_file, size, target_page)`: Creates a clickable image button.
- `main()`: Initializes and renders the homepage layout.

Author: Akshan Krithick
Date: March 17, 2025
"""
import os
import base64
import streamlit as st

st.set_page_config(page_title="FIFA 15 Style Homepage", layout="wide")

def get_base64(bin_file):
    """
    Encode a binary file into Base64 format.

    Args:
        bin_file (str): Path to the file.

    Returns:
        str: Base64-encoded string of the file content.
    """
    with open(bin_file, "rb") as file:
        return base64.b64encode(file.read()).decode()

def get_abs_path(relative_path):
    """
    Resolve the absolute path of an asset file.

    Args:
        relative_path (str): The relative path of the asset file.

    Returns:
        str | None: The absolute path of the asset if found, otherwise None.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    possible_paths = [
        os.path.join(current_dir, '..', 'assets', relative_path),
        os.path.join(current_dir, 'assets', relative_path),
        os.path.join(current_dir, relative_path),
        os.path.join('assets', relative_path)
    ]

    for path in possible_paths:
        abs_path = os.path.abspath(path)
        if os.path.exists(abs_path):
            return abs_path

    st.error(f"Could not find asset: {relative_path} in any known path.")
    return None

def set_background(image_path):
    """
    Set the background image for the Streamlit app.

    If the specified image is not found, displays an error message.

    Args:
        image_path (str): Absolute path to the background image.
    """
    if image_path is None:
        return

    if not os.path.exists(image_path):
        st.error(f"Background image not found: {image_path}")
        return

    img_b64 = get_base64(image_path)
    if img_b64:
        st.markdown(
            f"""
            <style>
                .stApp {{
                    background-image: url("data:image/jpeg;base64,{img_b64}");
                    background-size: cover;
                    background-position: center;
                    background-attachment: fixed;
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

def inject_custom_styles():
    """
    Inject custom CSS styles into the Streamlit app.

    Styles include:
    - Transparent buttons with hover effects
    - FIFA-style image buttons with animations
    """
    st.markdown(
        """
        <style>
            div.stButton > button {
                background-color: transparent !important;
                color: transparent !important;
                border: none !important;
                box-shadow: none !important;
                height: auto !important;
                padding: 5px !important;
                margin: 0 !important;
                cursor: pointer;
                width: 100%;
                margin-bottom: 0px;
            }
            div.stButton > button:hover {
                background-color: rgba(0, 0, 0, 0.1) !important;
            }
            .image-button {
                border-radius: 0;
                overflow: hidden;
                transition: transform 0.2s ease-in-out;
                margin: 5px;
                width: 100%;
                margin-top: 0px;
            }
            .image-button:hover {
                transform: scale(1.03);
                cursor: pointer;
            }
            .large-button { height: 300px; }
            .small-button { height: 150px; }
            .image-content {
                width: 100%;
                height: 100%;
                object-fit: fill;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

def display_image_button(image_file, size):
    """
    Display an image button.

    Args:
        image_file (str): Path to the image file.
        size (str): Button size, either 'large' or 'small'.
    """
    button_class = 'large-button' if size == 'large' else 'small-button'
    image_path = get_abs_path(image_file)

    if image_path is None:
        return

    img_b64 = get_base64(image_path)
    if img_b64:
        st.markdown(
            f"""
            <div class="image-button {button_class}">
                <img class="image-content" 
                     src="data:image/png;base64,{img_b64}" 
                     alt="{image_file}">
            </div>
            """,
            unsafe_allow_html=True,
        )

def fifa_button(image_file, size, target_page=None):
    """
    Create a clickable image button.

    When clicked, it navigates to a different page (if specified).

    Args:
        image_file (str): Path to the button image.
        size (str): Button size, either 'large' or 'small'.
        target_page (str, optional): Path to the target Streamlit page.
    """
    display_image_button(image_file, size)
    if st.button(label="", key=image_file):
        if target_page:
            st.switch_page(target_page)


def main():
    """
    Main function to initialize and render the FIFA-style homepage.

    - Sets the background image
    - Injects custom styles
    - Displays interactive FIFA-style buttons for navigation
    """
    set_background(get_abs_path('background.jpeg'))
    inject_custom_styles()

    row1_col1, row1_col2 = st.columns([1, 1])
    row2_col1, row2_col2 = st.columns([1, 1])

    with row1_col1:
        fifa_button('kickoff.png', 'large', 'pages/eda_salary.py')

    with row1_col2:
        fifa_button('stats.png', 'large', 'pages/player_plot.py')

    with row2_col1:
        fifa_button('similarity.png', 'small', 'pages/statmatch.py')

    with row2_col2:
        fifa_button('career.png', 'small', 'pages/the_negotiator.py')


if __name__ == "__main__":
    main()
