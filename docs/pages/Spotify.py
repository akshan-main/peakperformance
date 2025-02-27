import streamlit as st
import base64

# --- PAGE CONFIG ---
st.set_page_config(page_title="FIFA Spotify Playlists", layout="wide")

# --- REMOVE SIDEBAR ---
st.markdown(
    """
    <style>
        .css-1d391kg {display: none}
        .stApp {padding: 0; margin: 0;}
    </style>
    """,
    unsafe_allow_html=True,
)

# --- BACKGROUND IMAGE ---
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

# ✅ Back Button to Home (Returns to spotify.py)
if st.button("⬅️ Back to Home"):
    st.switch_page('spotify.py')

# ✅ Spotify Player Function (Global Player Across Pages)
def spotify_playlist(playlist_url):
    """Displays Spotify player and saves it globally in session."""
    if playlist_url:
        embed_url = playlist_url.replace("open.spotify.com/playlist/", "open.spotify.com/embed/playlist/")
        st.session_state["spotify_url"] = embed_url
        st.markdown(
            f"""
            <iframe src="{embed_url}" width="100%" height="380" frameborder="0" 
                    allowtransparency="true" allow="encrypted-media">
            </iframe>
            """,
            unsafe_allow_html=True,
        )

# ✅ Persist Playlist URL in Session
if "spotify_url" not in st.session_state:
    st.session_state["spotify_url"] = "https://open.spotify.com/embed/playlist/00i82lDzMDdiHWNjrIGAyw"

# --- TITLE ---
st.title("🎧 FIFA Radio")

# ✅ Playlist Selection (Updates Session State Only Here)
playlist_choice = st.radio(
    "FIFA Radio",
    ("FIFA 13", "FIFA 14", "FIFA 15"),
    horizontal=True
)

if playlist_choice == "FIFA 13":
    st.session_state["spotify_url"] = "https://open.spotify.com/embed/playlist/0GgDQMc3HSNukv5CVBDDFo"
elif playlist_choice == "FIFA 14":
    st.session_state["spotify_url"] = "https://open.spotify.com/embed/playlist/6uhKIwBpVEX9loDnk4iOcM"
elif playlist_choice == "FIFA 15":
    st.session_state["spotify_url"] = "https://open.spotify.com/embed/playlist/00i82lDzMDdiHWNjrIGAyw"

# ✅ Display Spotify Player Persistently
spotify_playlist(st.session_state["spotify_url"])
