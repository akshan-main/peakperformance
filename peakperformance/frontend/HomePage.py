import streamlit as st
import pandas as pd

# Load player data from CSV
@st.cache_data
def load_data():
    return pd.read_csv("../../dataset/cleaned_2023-24.csv")  # Change to your actual CSV file name

df = load_data()

#Title
st.markdown("<h1 style='text-align: center;'>Peak Performance</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>Search Player below</h4>", unsafe_allow_html=True)

# Search input
search_query = st.text_input("Enter player name:", "").strip()

# Filter players
#CHANGE THIS WHEN YOU HAVE TOP 100 PLAYERS
if search_query:
    filtered_df = df[df["player"].str.contains(search_query, case=False, na=False)]
else:
    # Show first few players when search is empty
    filtered_df = df.head(100)

# Handle player selection using session state
if 'selected_player' not in st.session_state:
    st.session_state.selected_player = None


# Display player details if one is selected
if st.session_state.selected_player:
    selected_player_data = df[df["player"] == st.session_state.selected_player].iloc[0]
    
    # Display detailed page for selected player
    st.markdown(f"<h2 style='text-align: center;'>Player Details</h2>", unsafe_allow_html=True)
    st.markdown(f"<h3>{selected_player_data['player']}</h3>", unsafe_allow_html=True)
    st.write(f"**Club:** {selected_player_data['squad']}")
    st.write(f"**Position:** {selected_player_data['pos']}")
    st.write(f"**Nationality:** {selected_player_data['nation']}")
    st.write(f"**Age:** {selected_player_data['age']}")
    
    # Add a button to go back to the player list
    if st.button("Back to Player List"):
        st.session_state.selected_player = None
        st.rerun()

else:
    # Display player cards as buttons
    if not filtered_df.empty:
        for _, row in filtered_df.iterrows():
            button_label = f"""\
{row["player"]}

{row["squad"]}
            """
            if st.button(button_label):
                st.session_state.selected_player = row["player"]
                st.rerun()
    else:
        st.write("No player found. Try another name.")