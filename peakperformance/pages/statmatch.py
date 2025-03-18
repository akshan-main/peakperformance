# pylint: disable=C0103  # Not a constant, dynamically generated HTML

"""
statmatch.py

This module implements a chatbot for finding similar players based on 
historical performance metrics. It uses cosine similarity on standardized 
player statistics to match players across different seasons.

Author: Akshan Krithick
Date: March 15, 2024
"""
import os
import re
import html
import base64
import unicodedata
from datetime import datetime
import streamlit as st
import pandas as pd
import numpy as np

def init_session_messages(session_state):
    """Initializes session messages without overwriting existing ones."""
    if "messages" not in session_state or not session_state.messages:
        session_state.messages = []
        session_state.messages.append({
            "role": "assistant",
            "content": """Hello! Ask for similar players! 
            For example: top 10 players similar to 2017 Cristiano Ronaldo 
            in the 2023 Bundesliga season under 25."""
        })
    return session_state.messages

def create_user_html(user_input, msg_time):
    """
    Returns HTML for user messages with proper escaping.

    Args:
        user_input (str): The query entered by the user.
        currenttime (str): The timestamp for the message.

    Returns:
        str: HTML string to display the user message.
    """
    query_escaped = html.escape(user_input)
    return f"""
        <div style='padding: 10px; margin: 10px 0; text-align: right; background-color: #075E54;
                    color: white; display: inline-block; border-radius: 13px 13px 0px 13px; max-width: 50%;
                    padding: 12px; float: right; clear: both; text-align: left;'>
                    {query_escaped}
                    <div style='font-size: 13px; color: #ece5dd; text-align: right; position: absolute; bottom: 12px; right: 10px; opacity: 0.5;'>
                        {msg_time}
                    </div>
        </div>
    """

def create_assistant_html(bot_output, msg_time):
    """Returns HTML for assistant's response."""
    final_html1 = """
        <div style='color: #25D366; font-weight: bold; font-size: 16px;'>
            ~ <span style='color: #25D366;'>STATMATCH</span>
        </div>
    """
    notepad_html = f"""
        <div style="
            background-color:  #273443;
            padding: 10px;
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
        {bot_output}
        <div style='font-size: 13px; color: #ece5dd; text-align: right; position: absolute; bottom: 8px; right: 10px; opacity: 0.5;'>
                {msg_time}
        </div>
        </div>
    """
    assistant_html = f"""
        <div style="margin-bottom: 25px; text-align: left; position: relative; max-width:60%;"> 
            {notepad_html}
        </div>
    """
    return assistant_html

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

currenttime = datetime.now().strftime("%H:%M")

def get_base64(bin_file):
    """Encodes a binary file into a base64 string.

    Args:
        bin_file (str): Path to the binary file.

    Returns:
        str: Base64-encoded string representation of the file,
        or None if the file is not found.
    """
    try:
        with open(bin_file, "rb") as file:
            return base64.b64encode(file.read()).decode()
    except FileNotFoundError:
        return None  # Return None instead of crashing in tests

def get_abs_path(relative_path):
    """
    Resolves the absolute path for a given relative path robustly by 
    traversing parent directories until finding the 'assets' directory.

    Args:
        relative_path (str): The relative path to the asset file.

    Returns:
        str | None: The absolute path to the asset if found, otherwise None.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Move upwards from current_dir until reaching root directory
    while True:
        potential_path = os.path.join(current_dir, 'assets', relative_path)
        if os.path.exists(potential_path):
            return potential_path

        # If current_dir is root or no further parent directory, stop searching
        parent_dir = os.path.dirname(current_dir)
        if current_dir == parent_dir:
            break  # Reached root directory without finding asset
        current_dir = parent_dir

    st.error(f"Could not find asset: {relative_path} in any known path.")
    return None

def load_background_image():
    """
    Loads the background image ('background.jpeg') and returns its base64-encoded string.

    Returns:
        str | None: Base64-encoded string of the background image, or None if not found.
    """
    bg_path = get_abs_path('background.jpeg')
    return get_base64(bg_path) if bg_path else None

bg_img = load_background_image()

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
    """Loads and preprocesses player data from CSV."""
    csv_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "dataset",
            "Ratings Combined", "thousandminsormore.csv"
        )
    )
    raw_df = pd.read_csv(csv_path)

    raw_df.drop_duplicates(subset=["PLAYER", "Season", "CLUB"], inplace=True)

    if "Goals Scored" in raw_df.columns:
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

def extract_years_and_results(user_input):
    """Extracts years and number of results from the user query.
    
    Args:
        user_input (str): The user's query input.

    Returns:
        Str : Parsed user query.
    """
    parsed = {"reference_season": None, "target_season": None, "num_results": 10}

    years = re.findall(r"\b20\d{2}\b", user_input)
    if years:
        parsed["reference_season"] = int(years[0])
        parsed["target_season"] = int(years[1]) if len(years) > 1 else int(years[0])

    match = re.search(r"top\s+(\d+)", user_input, flags=re.IGNORECASE)
    if match:
        parsed["num_results"] = int(match.group(1))

    return parsed

def extract_reference_player(user_input):
    """Extracts the reference player name from the user query.

    Args:
        user_input (str): The user's query input.

    Returns:
        str | None: Extracted reference player name or None if not found.
    """
    if "similar to" in user_input.lower():
        name_section = user_input.split("similar to", 1)[1]
        name_section = re.sub(r"\b20\d{2}\b", "", name_section, flags=re.IGNORECASE)
        name_section = re.split(
            r"\b(?:in(?: the)?|under|over|"
            r"\d+\s+or\s+younger|\d+\s+or\s+older|"
            r"season|league|exactly)\b",
            name_section,
            flags=re.IGNORECASE
        )[0].strip().strip(",")

        return clean_reference_player(name_section)
    return None

def clean_reference_player(name):
    """Cleans and validates a reference player name.

    Args:
        name (str | None): Extracted reference player name.

    Returns:
        str | None: Cleaned player name or None if invalid.
    """
    invalid_names = {"", "Top Players", "Top Similar Players", "Show Me Top Players"}
    return None if name in invalid_names else name.title()

def parse_user_query(user_input, data, demonym_mapping):
    """Parses the user input query into structured information for player similarity comparison.

    Args:
        user_input (str): The user's input query containing search parameters.
        data (pd.DataFrame): Dataset containing player information for validation.
        demonym_mapping (dict): Mapping of nationality adjectives to country names.

    Returns:
        dict: {
            "reference_player" (str | None): Extracted player name.
            "reference_season" (int | None): Season for the reference player.
            "target_season" (int | None): Season for finding similar players.
            "num_results" (int): Number of similar players requested.
            "league" (str | None): Extracted league name.
            "nationality" (str | None): Extracted nationality of the player.
            "age_filter" (tuple | None): Tuple with age condition and value.
        }
    """
    init_session_messages(st.session_state)

    user_html = create_user_html(user_input, datetime.now().strftime("%H:%M"))
    st.session_state.messages.append({"role": "user", "content": user_html})
    st.markdown(user_html, unsafe_allow_html=True)

    parsed = extract_years_and_results(user_input)
    parsed["reference_player"] = extract_reference_player(user_input)

    if not parsed["reference_player"] and not parsed["reference_season"]:
        error_msg = "Sorry, I didn't understand that. Try asking for similar players!"
        st.session_state.messages.append({"role": "assistant", "content": error_msg})
        st.markdown(error_msg, unsafe_allow_html=True)

        return {
            "reference_player": None,
            "reference_season": None,
            "target_season": None,
            "num_results": 10,
            "league": None,
            "nationality": None,
            "age_filter": None
        }

    leagues = ["Premier League", "La Liga", "Bundesliga", "Serie A", "Ligue 1", "PL"]
    parsed["league"] = next(
        (
            lg for lg in leagues
            if re.search(r'\b' + re.escape(lg) + r'\b', user_input, flags=re.IGNORECASE)
        ),
        None
    )

    if parsed["league"] == "PL":
        parsed["league"] = "Premier League"

    age_patterns = [
        (r"under\s+(\d+)", "under"),
        (r"over\s+(\d+)", "over"),
        (r"(\d+)\s+or\s+younger", "<="),
        (r"(\d+)\s+or\s+older", ">="),
        (r"exactly\s+(\d+)", "==")
    ]

    parsed["age_filter"] = next(
        (
            (operator, int(match.group(1)))
            for pattern, operator in age_patterns
            if (match := re.search(pattern, user_input, flags=re.IGNORECASE))
        ),
        None
    )

    countries = [c.lower() for c in data["nation"].unique()]
    parsed["nationality"] = next(
        (
            word.title()
            for word in user_input.lower().split()
            if word in countries
        ),
        None
    )
    if not parsed["nationality"]:
        parsed["nationality"] = next(
            (
                country
                for dem, country in demonym_mapping.items()
                if dem in user_input.lower()
            ),
            None
        )

    return parsed


def filter_candidates(player_data, normalized_data, user_query, metrics_list):
    """
    Filters potential candidate players based on similarity criteria.

    Args:
        player_data (pd.DataFrame): Raw player dataset.
        normalized_data (pd.DataFrame): Standardized dataset for computing similarity.
        user_query (dict): Extracted details from user input, including filters.
        metrics_list (list): List of selected performance metrics.

    Returns:
        tuple: (Filtered DataFrame, Normalized DataFrame, Reference Vector, Error Message)
    """
    ref_name = user_query.get("reference_player")
    ref_year = user_query.get("reference_season")
    target_year = user_query.get("target_season", ref_year)
    league = user_query.get("league")
    nationality = user_query.get("nationality")
    age_filter = user_query.get("age_filter")

    if not ref_name:
        return None, None, None, "Player not found. Please provide a valid player name."

    mask = (
        (player_data["Season"] == ref_year)
        & (
            player_data["PLAYER"].astype(str).str.lower().eq(ref_name.lower())
            | player_data["PLAYER"].astype(str).str.lower().str.contains(ref_name.lower(), na=False)
        )
    )

    if not mask.any():
        return None, None, None, f"Player **{ref_name}** in season **{ref_year}** was not found."

    ref_index = player_data[mask].index[0]
    reference_vector = np.asarray(
        normalized_data.loc[ref_index, metrics_list].values, dtype=np.float64
    )

    if np.linalg.norm(reference_vector) == 0 or np.all(np.isnan(reference_vector)):
        return None, None, None, (
            f"**{player_data.loc[ref_index, 'PLAYER']} ({ref_year})** "
            "has no values for the selected metrics."
        )

    filters = {
        "Season": player_data["Season"] == target_year,
        "League": (
            player_data["League"].str.lower().eq(league.lower()) if league else True
        ),
        "Nationality": (
            player_data["nation"].str.lower().eq(nationality.lower()) if nationality else True
        ),
        "Age": {
            "under": player_data["age"] < age_filter[1],
            "over": player_data["age"] > age_filter[1],
            "<=": player_data["age"] <= age_filter[1],
            ">=": player_data["age"] >= age_filter[1],
            "==": player_data["age"] == age_filter[1],
        }.get(age_filter[0], True) if age_filter else True,
    }

    candidate_mask = (
        filters["Season"] & filters["League"] & filters["Nationality"] & filters["Age"]
    )

    candidate_mask &= ~mask

    return player_data[candidate_mask], normalized_data[candidate_mask], reference_vector, None


def compute_similarity(
        reference_vector, candidate_data, normalized_candidates,
        metrics_list, num_results
    ):
    """
    Computes similarity scores between a reference player and candidate players.

    Args:
        reference_vector (np.array): Normalized vector of the reference player's statistics.
        candidate_data (pd.DataFrame): DataFrame containing potential candidate players.
        normalized_candidates (pd.DataFrame):
            Normalized candidate dataset for similarity computation.
        metrics_list (list): List of selected performance metrics.
        num_results (int): Number of top similar players to return.

    Returns:
        pd.DataFrame: A DataFrame containing the most similar players with similarity scores.
    """
    # Compute cosine similarity
    reference_norm = np.linalg.norm(reference_vector)
    candidate_matrix = normalized_candidates[metrics_list].values
    candidate_norms = np.linalg.norm(candidate_matrix, axis=1)
    dot_products = candidate_matrix.dot(reference_vector)
    similarity_scores = np.zeros(len(candidate_matrix))

    for i in range(len(candidate_matrix)):
        if candidate_norms[i] != 0:
            similarity_scores[i] = dot_products[i] / (reference_norm * candidate_norms[i])

    top_n = min(num_results, len(similarity_scores))
    top_indices = np.argsort(similarity_scores)[::-1][:top_n]
    similar_players = candidate_data.iloc[top_indices].copy()
    similar_players["Similarity"] = similarity_scores[top_indices]
    similar_players = similar_players[similar_players["Similarity"] > 0]

    return similar_players

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

init_session_messages(st.session_state)
query = st.chat_input("Enter your query:")
if query:
    parsed_query = parse_user_query(query, df, demonym_map)
    candidates_df, candidates_norm, ref_vector, error = filter_candidates(
        df, df_norm, parsed_query, selected_metrics
    )

    if error:
        st.session_state.messages.append({"role": "assistant", "content": error})
        st.markdown(error, unsafe_allow_html=True)
        st.stop()

    top_players = compute_similarity(
        ref_vector,
        candidates_df,
        candidates_norm,
        selected_metrics,
        parsed_query["num_results"]
    )

    if top_players.empty:
        result_text = "No players match the specified season and filters."
    else:
        result_lines = [
            f"<li>{row.PLAYER} ({row.CLUB}, {row.League}, age {row.age}) – "
            f"Similarity: {row.Similarity * 100:.1f}%</li>"
            for row in top_players.itertuples(index=False)
        ]

        result_text = f"<ol style='padding-left: 20px;'>{''.join(result_lines)}</ol>"

    final_html = create_assistant_html(result_text, currenttime)

    st.session_state.messages.append({"role": "assistant", "content": final_html})
    st.markdown(final_html, unsafe_allow_html=True)
