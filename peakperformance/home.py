import os
import base64
import streamlit as st

st.set_page_config(page_title="FIFA 15 Style Homepage", layout="wide")


def get_base64(bin_file):
    with open(bin_file, "rb") as file:
        return base64.b64encode(file.read()).decode()

def get_abs_path(relative_path):
    """Ensure the correct path to assets in both local and deployed environments."""
    
    # Check if the relative path works locally
    if os.path.exists(relative_path):
        return relative_path  

    # If running inside Streamlit deployment, adjust path
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get script directory
    return os.path.join(base_dir, "..", "assets", relative_path)


def set_background(image_path):
    if not os.path.exists(image_path):
        st.error(f"Background image not found: {image_path}")
        return

    img_b64 = get_base64(image_path)
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
            .large-button { height: 450px; }
            .small-button { height: 250px; }
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
    button_class = 'large-button' if size == 'large' else 'small-button'
    image_path = get_abs_path(image_file)

    if not os.path.exists(image_path):
        st.error(f"Image not found: {image_path}")
        return

    with open(image_path, "rb") as img_file:
        img_b64 = base64.b64encode(img_file.read()).decode()

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
    display_image_button(image_file, size)
    if st.button(label="", key=image_file):
        if target_page:
            st.switch_page(target_page)


def main():
    set_background(get_abs_path('background.jpeg'))
    inject_custom_styles()

    row1_col1, row1_col2 = st.columns([1, 1])
    row2_col1, row2_col2 = st.columns([1, 1])

    with row1_col1:
        fifa_button('kickoff.png', 'large', 'pages/1_salary.py')

    with row1_col2:
        fifa_button('stats.jpeg', 'large', 'pages/2_plot.py')

    with row2_col1:
        fifa_button('similarity.jpeg', 'small', 'pages/3_chatbot.py')

    with row2_col2:
        fifa_button('career.jpg', 'small', 'pages/4_RLmodel.py')


if __name__ == "__main__":
    main()
