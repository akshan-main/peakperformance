import streamlit as st


st.markdown("<h1 style='text-align: center;'>Peak Performance</h1>", unsafe_allow_html=True)

st.markdown("<h4 style='text-align: center;'>Search Player below</h4>", unsafe_allow_html=True)


# Sample player data
players = {
    "Lionel Messi": {"Club": "Inter Miami", "Position": "Forward", "Nationality": "Argentina", "Age": 36},
    "Cristiano Ronaldo": {"Club": "Al-Nassr", "Position": "Forward", "Nationality": "Portugal", "Age": 39},
    "Kylian Mbappe": {"Club": "Paris Saint-Germain", "Position": "Forward", "Nationality": "France", "Age": 25},
    "Erling Haaland": {"Club": "Manchester City", "Position": "Striker", "Nationality": "Norway", "Age": 23},
    "Kevin De Bruyne": {"Club": "Manchester City", "Position": "Midfielder", "Nationality": "Belgium", "Age": 32},
    "Neymar Jr": {"Club": "Al-Hilal", "Position": "Forward", "Nationality": "Brazil", "Age": 32},
    "Vinicius Jr": {"Club": "Real Madrid", "Position": "Winger", "Nationality": "Brazil", "Age": 23},
    "Mohamed Salah": {"Club": "Liverpool", "Position": "Winger", "Nationality": "Egypt", "Age": 31},
    "Jude Bellingham": {"Club": "Real Madrid", "Position": "Midfielder", "Nationality": "England", "Age": 20},
    "Harry Kane": {"Club": "Bayern Munich", "Position": "Striker", "Nationality": "England", "Age": 30},
}

search_query = st.text_input("Enter player name:", "").strip()

# Filter players
if search_query:
    filtered_players = {name: details for name, details in players.items() if search_query.lower() in name.lower()}
else:
    # Show first few players when search is empty
    filtered_players = dict(list(players.items())[:5])  # Display first 5 players

# Display player cards
if filtered_players:
    for name, details in filtered_players.items():
        st.markdown(f"""
        <div style="border:2px solid #ccc; padding: 10px; border-radius: 10px; margin-bottom: 10px;">
            <h3 style="color: #007BFF;">{name}</h3>
            <p><strong>Club:</strong> {details['Club']}</p>
            <p><strong>Position:</strong> {details['Position']}</p>
            <p><strong>Nationality:</strong> {details['Nationality']}</p>
            <p><strong>Age:</strong> {details['Age']}</p>
        </div>
        """, unsafe_allow_html=True)
else:
    st.write("No player found. Try another name.")