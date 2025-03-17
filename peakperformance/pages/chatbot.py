# pylint: disable=C0103  # Not a constant, dynamically generated HTML

"""
testingchatbotnow.py

This module implements a chatbot for finding similar players based on 
historical performance metrics. It uses cosine similarity on standardized 
player statistics to match players across different seasons.

Author: Akshan Krithick
Date: March 15, 2024
"""
import re
import base64
import unicodedata
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

currenttime = datetime.now().strftime("%H:%M")

def get_base64(bin_file):
    """Encodes a binary file into a base64 string.

    Args:
        bin_file (str): Path to the binary file.

    Returns:
        str: Base64-encoded string representation of the file.
    """
    with open(bin_file, "rb") as file:
        return base64.b64encode(file.read()).decode()

bg_img = get_base64('../../background.jpeg')
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
st.title("STATMATCH")

def remove_accents(text):
    """Removes accents from a given string.

    Converts accented characters to their closest ASCII equivalent.

    Args:
        text (str): The input string containing accented characters.

    Returns:
        str: The string with accents removed.
    """
    if isinstance(text, str):
        return unicodedata.normalize('NFKD', text).encode('ASCII', 'ignore').decode('utf-8')
    return text

@st.cache_data
def load_data():
    """Loads and preprocesses player data."""
    raw_df = pd.read_csv("../../datasets/Ratings Combined/thousandminsormore.csv")
    raw_df.drop_duplicates(subset=["PLAYER", "Season", "CLUB"], inplace=True)
    if "Goals Scored" in df.columns:
        raw_df.drop(columns=["Goals Scored"], inplace=True)
    for col in ["rk", "Rating", "GROSS P/W (EUR)", "GROSS P/Y (EURTAX)", "GROSS P/Y (EUR)"]:
        if col in raw_df.columns:
            raw_df.drop(columns=[col], inplace=True)
    raw_df["PLAYER"] = raw_df["PLAYER"].apply(remove_accents)

    raw_metrics_cols = [
        c for c in raw_df.columns
        if c not in [
            "PLAYER", "nation", "pos", "CLUB",
            "League", "age", "born", "Season"
        ]
    ]
    raw_df_norm = raw_df.copy()
    for col in raw_metrics_cols:
        mean = raw_df[col].mean()
        std = raw_df[col].std(ddof=0)
        raw_df_norm[col] = (raw_df[col] - mean) / std if std != 0 else raw_df[col]

    return raw_df, raw_df_norm, raw_metrics_cols

df, df_norm, metrics_cols = load_data()

st.sidebar.subheader("Select Performance Metrics")
use_all = st.sidebar.checkbox("Use all metrics", value=True)
if use_all:
    selected_metrics = metrics_cols
else:
    selected_metrics = st.sidebar.multiselect(
        "Choose metrics for similarity:", metrics_cols, default=metrics_cols[2:6]
    )
    if not selected_metrics:
        st.sidebar.warning("Select at least one metric or use all metrics.")
        st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant",
        "content": """Hello! Ask for similar players! 
        For example: top 10 players similar to 2017 Cristiano Ronaldo 
        in the 2023 Bundesliga season under 25."""
    })

for msg in st.session_state.messages:
    st.markdown(msg["content"], unsafe_allow_html=True)


query = st.chat_input("Enter your query:")
if query:
    user_html = f"""
        <div style='border-radius: 10px; padding: 10px; margin: 10px 0; text-align: right;
                    background-color: #075E54; color: white; display: inline-block;
                    border-radius: 13px 13px 0px 13px; max-width: 50%; padding: 12px; float: right; clear: both; text-align: left;'>
                    {query}
                    <div style='font-size: 12px; color: #ece5dd; text-align: right; position: absolute; bottom: 10px; right: 10px; opacity: 0.5;'>
                            {currenttime}
        </div>
    """
    st.session_state.messages.append({"role": "user", "content": user_html})
    st.markdown(user_html, unsafe_allow_html=True)

    parsed = {
        "reference_player": None,
        "reference_season": None,
        "target_season": None,
        "num_results": 10,
        "league": None,
        "nationality": None,
        "age_filter": None
    }

    years = re.findall(r"\b20\d{2}\b", query)
    if years:
        parsed["reference_season"] = int(years[0])
        parsed["target_season"] = int(years[1]) if len(years) > 1 else int(years[0])

    m = re.search(r"top\s+(\d+)", query, flags=re.IGNORECASE)
    if m:
        parsed["num_results"] = int(m.group(1))

    name_section = query
    if "similar to" in query.lower():
        name_section = query.split("similar to", 1)[1]
    if " in the" in name_section.lower():
        name_section = name_section.split(" in the", 1)[0]
    name_section = re.sub(r"\b20\d{2}\b", "", name_section, flags=re.IGNORECASE)
    name_section = name_section.replace("season", "").strip().strip(",")
    parsed["reference_player"] = name_section if name_section else None

    leagues = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "EPL"]
    for lg in leagues:
        if lg.lower() in query.lower():
            parsed["league"] = "Premier League" if lg.lower() == "epl" else lg
            break

    countries = [c.lower() for c in df["nation"].unique()]
    nationality_words = query.lower().split()
    for word in nationality_words:
        if word in countries:
            parsed["nationality"] = word.title()
            break

    demonym_map = {
        "english": "England",
        "french": "France",
        "spanish": "Spain",
        "german": "Germany",
        "italian": "Italy",
        "dutch": "Netherlands",
        "portuguese": "Portugal",
        "brazilian": "Brazil",
        "argentinian": "Argentina",
        "argentine": "Argentina",
        "belgian": "Belgium",
        "uruguayan": "Uruguay",
        "ivorian": "Ivory Coast",
        "croatian": "Croatia",
        "senegalese": "Senegal",
        "polish": "Poland",
        "egyptian": "Egypt",
        "nigerian": "Nigeria",
        "cameroonian": "Cameroon",
        "danish": "Denmark",
        "swedish": "Sweden",
        "norwegian": "Norway",
        "japanese": "Japan",
        "korean": "South Korea",
        "australian": "Australia",
        "american": "United States",
        "canadian": "Canada",
        "mexican": "Mexico",
        "chilean": "Chile",
        "colombian": "Colombia",
        "peruvian": "Peru",
        "venezuelan": "Venezuela",
        "ecuadorian": "Ecuador",
        "paraguayan": "Paraguay",
        "bolivian": "Bolivia",
        "costa rican": "Costa Rica",
        "salvadoran": "El Salvador",
        "guatemalan": "Guatemala",
        "honduran": "Honduras",
        "panamanian": "Panama",
        "nicaraguan": "Nicaragua",
        "cuban": "Cuba",
        "puerto rican": "Puerto Rico",
        "dominican": "Dominican Republic",
        "haitian": "Haiti",
        "jamaican": "Jamaica",
        "trinidadian": "Trinidad and Tobago",
        "barbadian": "Barbados",
        "guyanese": "Guyana",
        "surinamese": "Suriname",
        "ghanian": "Ghana",
        "kenyan": "Kenya",
        "south african": "South Africa",
        "malian": "Mali",
        "greek": "Greece",
        "turkish": "Turkey",
        "russian": "Russia",
        "ukrainian": "Ukraine",
        "swiss": "Switzerland",
        "austrian": "Austria",
        "czech": "Czech Republic",
        "slovak": "Slovakia",
        "hungarian": "Hungary",
        "bulgarian": "Bulgaria",
        "romanian": "Romania",
        "serbian": "Serbia"
    }

    for dem, country in demonym_map.items():
        if dem in query.lower():
            parsed["nationality"] = country
            break

    if "under" in query.lower():
        m = re.search(r"under\s+(\d+)", query, flags=re.IGNORECASE)
        if m:
            parsed["age_filter"] = ("under", int(m.group(1)))
    if "over" in query.lower():
        m = re.search(r"over\s+(\d+)", query, flags=re.IGNORECASE)
        if m:
            parsed["age_filter"] = ("over", int(m.group(1)))
    if "or younger" in query.lower():
        m = re.search(r"(\d+)\s+or\s+younger", query, flags=re.IGNORECASE)
        if m:
            parsed["age_filter"] = ("<=", int(m.group(1)))
    if "or older" in query.lower():
        m = re.search(r"(\d+)\s+or\s+older", query, flags=re.IGNORECASE)
        if m:
            parsed["age_filter"] = (">=", int(m.group(1)))

    ref_name = parsed.get("reference_player")
    ref_year = parsed.get("reference_season")
    target_year = parsed.get("target_season") or ref_year
    num_results = parsed.get("num_results", 10)
    league_filter = parsed.get("league")
    nationality_filter = parsed.get("nationality")
    age_filter = parsed.get("age_filter")

    error_message = ""
    if not ref_name or not ref_year:
        error_message = """Sorry, I couldn't identify the reference player
         and season from your query."""
    else:
        mask = (df["Season"] == ref_year) & (df["PLAYER"].str.lower() == ref_name.lower())
        if not mask.any():
            mask = (
                (df["Season"] == ref_year) &
                (df["PLAYER"].str.lower().str.contains(ref_name.lower()))
            )
        if not mask.any():
            error_message = f"Player **{ref_name}** in season **{ref_year}** was not found."

    if error_message:
        with st.chat_message("assistant"):
            st.markdown(error_message)
        st.session_state.messages.append({"role": "assistant", "content": error_message})
        st.stop()

    ref_index = df[mask].index[0]
    ref_vector = df_norm.loc[ref_index, selected_metrics].values
    if np.linalg.norm(ref_vector) == 0:
        msg = (
            f"**{df.loc[ref_index, 'PLAYER']} ({ref_year})** "
            "has no values for the selected metrics."
        )
        with st.chat_message("assistant"):
            st.markdown(msg)
        st.session_state.messages.append({"role": "assistant", "content": msg})
        st.stop()

    target_mask = df["Season"] == target_year
    if league_filter:
        target_mask &= df["League"].str.lower().eq(league_filter.lower())
    if nationality_filter:
        target_mask &= df["nation"].str.lower().eq(nationality_filter.lower())
    if age_filter:
        flt_type, age_val = age_filter if isinstance(age_filter, (tuple, list)) else (None, None)
        if flt_type in ["under", "<"]:
            target_mask &= df["age"] < age_val
        elif flt_type in ["over", ">"]:
            target_mask &= df["age"] > age_val
        elif flt_type in ["<=", "or younger"]:
            target_mask &= df["age"] <= age_val
        elif flt_type in [">=", "or older"]:
            target_mask &= df["age"] >= age_val

    target_mask &= ~((df["Season"] == ref_year) & (df["PLAYER"].str.lower() == ref_name.lower()))

    candidates_df = df[target_mask]
    candidates_norm = df_norm[target_mask]

    if candidates_df.empty:
        result_text = "No players match the specified season and filters."
    else:
        A = ref_vector
        A_norm = np.linalg.norm(A)
        B_matrix = candidates_norm[selected_metrics].values
        B_norms = np.linalg.norm(B_matrix, axis=1)
        dot_prods = B_matrix.dot(A)
        cosine_sim = np.zeros(len(B_matrix))
        for i in range(len(B_matrix)):
            if B_norms[i] != 0:
                cosine_sim[i] = dot_prods[i] / (A_norm * B_norms[i])
        top_n = min(num_results, len(cosine_sim))
        top_idx = np.argsort(cosine_sim)[::-1][:top_n]
        top_players = candidates_df.iloc[top_idx].copy()
        top_players["Similarity"] = cosine_sim[top_idx]
        top_players = top_players[top_players["Similarity"] > 0]
        # Ensure result lines are formatted consistently
        result_lines = [
            (
                f"<li>{row.PLAYER} ({row.CLUB}, {row.League}, age {row.age}) – "
                f"Similarity: {row.Similarity * 100:.1f}%</li>"
            )
            for row in top_players.itertuples(index=False)
        ]

        # Wrap everything in an ordered list `<ol>`
        result_text = f"<ol style='padding-left: 20px;'>{''.join(result_lines)}</ol>"

        final_html1 = """
            <div style='color: #25D366; font-weight: bold; font-size: 16px;'>
                ~ <span style='color: #25D366;'>STATMATCH</span>
            </div>
        """
        notepad_html = f"""
            <div style="
                background-color:  #273443;
                padding: 8px;
                border-radius: 13px 13px 13px 0px;
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
                font-size: 14px;
                line-height: 1.6;
                color: white;
                white-space: pre-wrap;
                box-shadow: 4px 4px 10px rgba(0,0,0,0.1);
                margin-top: 10px;
                margin-bottom: 10px;
                overflow: hidden;
            ">
            {final_html1}
            {result_text}
            <div style='font-size: 12px; color: #ece5dd; text-align: right; position: absolute; bottom: 10px; right: 10px; opacity: 0.5;'>
                    {currenttime}
            </div>
            </div>
            """

        final_html = f"""
            <div style="margin-bottom: 25px; text-align: left; position: relative; max-width:60%;"> 
                {notepad_html}
            </div>
        """

        st.session_state.messages.append({"role": "assistant", "content": final_html})
        st.markdown(final_html, unsafe_allow_html=True)
