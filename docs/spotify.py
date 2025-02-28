import streamlit as st
import base64
import os

st.set_page_config(page_title="FIFA 15 Style Homepage", layout="wide")

def get_base64(bin_file):
    with open(bin_file, "rb") as file:
        return base64.b64encode(file.read()).decode()

bg_img = get_base64('background.jpeg')
st.markdown(
    f"""
    <style>
        .stApp {{
            background-image: url("data:image/jpeg;base64,{bg_img}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
        div.stButton > button {
            background-color: transparent !important; /* Fully Transparent */
            color: transparent !important;           /* Text Invisible */
            border: none !important;                 /* No Border */
            box-shadow: none !important;             /* No Shadow */
            height: auto !important;
            padding: 5px !important;
            margin: 0 !important;
            cursor: pointer;
            width: 100%;
            margin-bottom: 0px;
        }
        div.stButton > button:hover {
            background-color: rgba(0, 0, 0, 0.1) !important; /* Slight hover feedback */
        }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <style>
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

def fifa_button(image_file, size, target_page=None):
    """Displays an image and uses a transparent button for click action."""
    button_class = 'large-button' if size == 'large' else 'small-button'
    
    if os.path.exists(image_file):
        with open(image_file, "rb") as img_file:
            img_bytes = img_file.read()
            img_base64 = base64.b64encode(img_bytes).decode()

        st.markdown(
            f"""
            <div class="image-button {button_class}">
                <img class="image-content" 
                     src="data:image/png;base64,{img_base64}" 
                     alt="{image_file}">
            </div>
            """,
            unsafe_allow_html=True,
        )

        if st.button(label="", key=image_file):
            if target_page:
                st.switch_page(target_page)
    else:
        st.error(f"Image '{image_file}' not found.")

row1_col1, row1_col2 = st.columns([1, 1])
row2_col1, row2_col2 = st.columns([1, 1])

with row1_col1:
    fifa_button('kickoff.jpeg', 'large')

with row1_col2:
    fifa_button('luka.jpeg', 'large')

with row2_col1:
    fifa_button('playerstats.jpeg', 'small')

with row2_col2:
    fifa_button('nowavailable.jpeg', 'small', 'pages/Spotify.py')


