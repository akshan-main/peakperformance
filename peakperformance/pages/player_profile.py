"""
Football Player Profile Page

This module sets up a Streamlit page for displaying football player profiles.
It uses data from a CSV file and APIs from TheSportsDB.

Author: Balaji Boopal
Date: March 16, 2025
"""

import io
from pathlib import Path

import streamlit as st
import requests
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Global Constants
CSV_PATH: Path = (
    Path(__file__).resolve().parent.parent.parent
    / "dataset"
    / "Ratings Combined"
    / "player_data_with_predictions.csv"
)
API_KEY: str = "3"  # Free API key from TheSportsDB
PLAYER_API_URL: str = f"https://www.thesportsdb.com/api/v1/json/{API_KEY}/searchplayers.php"


def set_theme() -> None:
    """
    Sets the theme for the Streamlit page.
    - Main page: dark grey gradient background.
    - Sidebar: darker background with near-white text.
    - Main text: near-white for readability.
    - Centers the content.
    
    Returns:
        None
    """
    st.markdown(
        """
        <style>
            /* Main page background: dark grey gradient */
            .stApp {
                background: #15202B !important;
                display: flex;
                justify-content: center;
                align-items: center;
            }
            
            /* Center main content and set width */
            .main-container {
                max-width: 75% !important;
                margin: auto;
                padding-top: 20px;
            }
            
            /* Set default text color to near-white for readability */
            html, body, [class*="css"] {
                color: #EEE !important;
            }
            
            /* Sidebar styling: dark background with near-white text */
            section[data-testid="stSidebar"] {
                background-color: #37444D !important;
                width: 18% !important;
                color: #EEE !important;
            }
            
            /* Center text and tables */
            .stMarkdown, .stText, .stDataFrame, .stMetric {
                text-align: center !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def load_player_data(file_path: str) -> pd.DataFrame:
    """
    Loads player data from a CSV file.

    Args:
        file_path (str): The path to the CSV file containing player data.

    Returns:
        pd.DataFrame: A DataFrame containing the player data.
                      Returns an empty DataFrame if the file is not found.
    """
    try:
        return pd.read_csv(file_path)
    except FileNotFoundError:
        st.error("⚠️ Player data file not found.")
        return pd.DataFrame()


def fetch_player_data(player_name: str) -> dict:
    """
    Fetches player data from TheSportsDB API.

    Args:
        player_name (str): The name of the player to search for.

    Returns:
        dict: A dictionary containing the player's data if found,
              otherwise an empty dictionary.
    """
    try:
        response = requests.get(f"{PLAYER_API_URL}?p={player_name}", timeout=5)
    except requests.RequestException as exc:
        st.error(f"Request failed: {exc}")
        return {}

    if response.status_code == 200:
        data = response.json()
        if data.get("player"):
            return data["player"][0]
    return {}


def display_player_header(player_data: dict) -> None:
    """
    Displays the player header including photo, name, and team name.

    Args:
        player_data (dict): A dictionary containing the player's data.

    Returns:
        None
    """
    player_name = player_data.get("strPlayer", "Unknown Player")
    team_name = player_data.get("strTeam", "Unknown")
    photo_url = player_data.get("strThumb", "https://via.placeholder.com/150")

    st.markdown(
        f"""
        <div class='player-card'>
            <div class='player-header'>
                <img src='{photo_url}' class='player-photo' style="width: 300px;
                height: auto; border-radius: 10px;">
                <div>
                    <h2>{player_name}</h2>
                    <p><strong>{team_name}</strong></p>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_player_info(player_data: dict) -> None:
    """
    Displays the player's nationality and position information.

    Args:
        player_data (dict): A dictionary containing the player's data.

    Returns:
        None
    """
    nationality = player_data.get("strNationality", "Unknown")
    position = player_data.get("strPosition", "Unknown")

    col1, col2 = st.columns(2)
    col1.metric("Nationality", nationality)
    col2.metric("Position", position)


def get_rating_trend(player_name: str, df: pd.DataFrame) -> pd.DataFrame:
    """
    Fetches the rating trend for a given player from the DataFrame.

    Args:
        player_name (str): The name of the player.
        df (pd.DataFrame): The DataFrame containing player ratings data.

    Returns:
        pd.DataFrame: A DataFrame with 'season' and 'Rating' columns for the player.
                      Returns None if the player data is not found.
    """
    player_data = df[df["player"].str.lower() == player_name.lower()]
    if player_data.empty:
        return None
    return player_data[["season", "Rating"]].dropna()


def plot_rating_trend(player_name: str, df: pd.DataFrame) -> io.BytesIO:
    """
    Plots the rating trend for a player with separate colors for actual and predicted ratings.

    Args:
        player_name (str): The name of the player.
        df (pd.DataFrame): The DataFrame containing player ratings data.

    Returns:
        io.BytesIO: A BytesIO object containing the PNG image of the plot.
                    Returns None if no data is found for the player.
    """
    player_data = df[df["player"].str.lower() == player_name.lower()]
    if player_data.empty:
        return None

    # Create a figure and set its size
    fig, ax = plt.subplots(figsize=(10, 5))

    # Sort player data by season
    player_data = player_data.sort_values("season")

    # Separate actual and predicted ratings
    actual_ratings = player_data.dropna(subset=["Rating"])
    predicted_ratings = player_data.dropna(subset=["predicted_rating"])

    # Plot actual ratings if available
    if not actual_ratings.empty:
        ax.plot(
            actual_ratings["season"],
            actual_ratings["Rating"],
            marker="o",
            linestyle="-",
            color="white",
            label="Actual Rating",
            markersize=8,
            linewidth=2,
        )

    # Plot predicted ratings if available
    if not predicted_ratings.empty:
        ax.plot(
            predicted_ratings["season"],
            predicted_ratings["predicted_rating"],
            marker="o",
            linestyle="dashed",
            color="red",
            label="Predicted Rating",
            markersize=8,
            linewidth=2,
        )

    # Customize plot ticks and labels
    ax.set_xticks(
        np.arange(
            int(player_data["season"].min()),
            int(player_data["season"].max()) + 2,
            1,
        )
    )
    ax.tick_params(axis='x', labelsize=12, colors='white')  # Change tick label color to white
    ax.tick_params(axis='y', labelsize=12, colors='white')  # Change tick label color to white
    ax.set_xlabel("Season", fontsize=14, color='white')  # Change x-axis label color to white
    ax.set_ylabel("Player Rating", fontsize=14, color='white')  # Change y-axis label color to white
    ax.set_title(f"Player Rating Trend: {player_name}", fontsize=16, color='white')
    ax.legend(fontsize=12, facecolor='#37444D', edgecolor='white')  # Legend background and border

    # Change the background colors
    ax.set_facecolor('#37444D')  # Set plot area background to #37444D
    fig.patch.set_facecolor('#37444D')  # Set figure background to #37444D

    # Add gridlines with a custom style for better visibility
    ax.grid(True, linestyle="--", alpha=0.6, color='white')  # Gridlines in white

    # Save the plot to a buffer
    img_buffer = io.BytesIO()
    plt.savefig(img_buffer, format="png", bbox_inches="tight", facecolor='#37444D')
    plt.close()
    img_buffer.seek(0)
    return img_buffer


def get_latest_season_stats(player_name: str, df: pd.DataFrame) -> dict:
    """
    Fetches the latest season statistics for a player from the DataFrame.

    Args:
        player_name (str): The name of the player.
        df (pd.DataFrame): The DataFrame containing player statistics.

    Returns:
        dict: A dictionary with the latest season stats including season, goals,
              assists, matches played, and current rating.
              Returns None if the player's data is not found.
    """
    player_data = df[df["player"].str.lower() == player_name.lower()]
    if player_data.empty:
        return None

    latest_season = player_data["season"].max()
    latest_data = player_data[player_data["season"] == latest_season].iloc[0]

    return {
        "season": latest_season,
        "goals": latest_data.get("Goals", "N/A"),
        "assists": latest_data.get("Assists", "N/A"),
        "matches": latest_data.get("Matches Played", "N/A"),
        "current_rating": latest_data.get("Rating", "N/A"),
    }


def main() -> None:
    """
    Main function to run the Streamlit app for the football player profile page.

    Loads the player data, fetches and displays player information,
    and plots the rating trend.

    Returns:
        None
    """
    df = load_player_data(str(CSV_PATH))

    with st.container():
        st.markdown(
            "<h1 style='text-align: center;'>⚽ Football Player Profile</h1>",
            unsafe_allow_html=True,
        )

    if df.empty:
        st.error("No player data available.")
        return

    available_players = sorted(df["player"].dropna().unique())
    selected_player = st.selectbox("🔍 Select a player:", available_players)

    if selected_player:
        player_data = fetch_player_data(selected_player)

        if player_data:
            display_player_header(player_data)
            display_player_info(player_data)

            st.markdown("<h3>League Performance</h3>", unsafe_allow_html=True)
            current_year = 2022
            current_season_data = df[
                (df["season"] == current_year)
                & (df["player"].str.lower() == selected_player.lower())
            ]

            if not current_season_data.empty:
                latest_data = current_season_data.iloc[0]
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Matches Played", latest_data.get(
                    "Matches Played", "N/A"))
                col2.metric("Goals", latest_data.get("Goals", "N/A"))
                col3.metric("Assists", latest_data.get("Assists", "N/A"))
                col4.metric("Current Rating", latest_data.get("Rating", "N/A"))
                col5, col6 = st.columns(2)
                col5.metric("Age", latest_data.get("age", "N/A"))
                col6.metric("Expected Goals", latest_data.get(
                    "Expected Goals", "N/A"))
            else:
                col1, col2 = st.columns(2)
                col1.metric("Matches Played", "N/A")
                col2.metric("⭐ Current Rating", "N/A")

            stats = get_latest_season_stats(selected_player, df)
            if stats:
                rating_plot = plot_rating_trend(selected_player, df)
                if rating_plot:
                    st.image(
                        rating_plot, caption=f"{selected_player} - Rating Trend")
            else:
                st.error("❌ No stats available for this player.")
        else:
            st.error("❌ Player not found. Try another name.")


if __name__ == "__main__":
    set_theme()
    main()
