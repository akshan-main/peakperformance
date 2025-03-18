"""
Player Performance Metrics Radar Chart Visualization

This Streamlit application loads player ratings and salary data from a CSV file, 
allowing users to interactively select players, seasons, and performance metrics 
to visualize comparisons using radar charts.

Features:
- Loads and processes player ratings from a dataset.
- Provides an interactive sidebar for player and metric selection.
- Generates a radar chart for visualizing performance metrics.
- Uses Plotly for dynamic data visualization.

Modules:
- os: For handling file paths dynamically.
- streamlit: For creating an interactive web application.
- pandas: For data manipulation and processing.
- plotly.graph_objects: For creating radar charts.

Author: Joshua Son
Date: March 16, 2025
"""

import os
import streamlit as st
import pandas as pd
import plotly.graph_objects as go


def load_data(filepath=None):
    """
    Load player ratings data from a CSV file.

    Args:
        filepath (str, optional): Path to the CSV file. If None, loads from default location.

    Returns:
        tuple: (pd.DataFrame, list) Processed DataFrame and available performance metrics.
    """
    if filepath is None:
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        dataset_path = os.path.join(
            project_root, "..", "dataset", "Ratings Combined", "playerratingssalaries_100mins.csv"
        )
    else:
        dataset_path = filepath

    player_df = pd.read_csv(dataset_path)

    exclude_cols = {
        "rk", "PLAYER", "nation", "pos", "CLUB", "League", "age", "born", "Season",
        "GROSS P/W (EUR)", "GROSS P/Y (EUR)", "GROSS P/Y (EURTAX)"
    }
    metrics = [col for col in player_df.columns if col not in exclude_cols]

    rename_map = {col: col.replace("%", "Percent").replace("p 90", " per 90") for col in metrics}
    player_df.rename(columns=rename_map, inplace=True)

    metrics = [rename_map[col] for col in metrics]
    player_df[metrics] = player_df[metrics].apply(pd.to_numeric, errors="coerce")

    return player_df, metrics


def compute_scaled_values(data_df, metrics_list, player_series):
    """
    Compute scaled values for radar chart.

    Args:
        data_df (pd.DataFrame): Full dataset.
        metrics_list (list): Selected performance metrics.
        player_series (pd.Series): Row corresponding to a selected player.

    Returns:
        dict: Contains scaled values and hover text.
    """
    max_values = data_df[metrics_list].max().replace(0, 1e-5)
    raw_values = player_series[metrics_list].astype(float)
    scaled_values = (raw_values / max_values) * 100

    values = scaled_values.fillna(0).tolist()
    values.append(values[0])  # Close radar chart loop

    hover_texts = [
        (
            f"<b>{metric}</b><br>"
            f"Raw: {raw:.2f}<br>"
            f"Scaled: {scaled:.2f}%<br>"
            f"Max: {max_values[metric]:.2f}<br>"
            f"<b>{player_series['PLAYER']} {player_series['Season']}</b>"
        )
        for metric, raw, scaled in zip(metrics_list, raw_values, values[:-1])
    ]


    return {"scaled_values": values, "hover_texts": hover_texts}


def generate_radar_chart(data_df, selected_players_df, metrics_list):
    """
    Generate a radar chart for player comparisons.

    Args:
        data_df (pd.DataFrame): Full dataset.
        selected_players_df (pd.DataFrame): DataFrame of selected players.
        metrics_list (list): Selected performance metrics.

    Returns:
        go.Figure: Radar chart figure.
    """
    radar_fig = go.Figure()

    for _, player_data in selected_players_df.iterrows():
        computed_values = compute_scaled_values(data_df, metrics_list, player_data)
        values = computed_values["scaled_values"]
        hover_texts = computed_values["hover_texts"]

        radar_fig.add_trace(go.Scatterpolar(
            r=values,
            theta=metrics_list + [metrics_list[0]],
            fill="toself",
            name=f"{player_data['PLAYER']} {player_data['Season']}",
            line={"width": 3},
            mode="markers+lines",
            marker={"size": 8, "opacity": 0.8},
            hovertemplate="<br>".join(["%{text}"]),
            text=hover_texts,
        ))

    radar_fig.update_layout(
        margin={"l": 130, "r": 100, "t": 100, "b": 100},
        width=950,
        height=850,
        autosize=False,
        polar={
            "radialaxis": {
                "visible": True,
                "range": [0, 100],
                "showticklabels": True,
                "ticks": "outside",
                "linewidth": 1,
                "tickfont": {"color": "black", "size": 12},
            }
        },
        showlegend=True,
        title={"text": "<b>Player Performance Radar</b>", "font": {"size": 24}},
        font={"size": 12},
    )

    return radar_fig


# Streamlit UI
df_main, available_metrics = load_data()
st.sidebar.title("Filters")

num_selected_players = st.sidebar.slider("Number of Player Selections", 1, 10, 2)
selected_players = []

for idx in range(num_selected_players):
    selected_player = st.sidebar.selectbox(
        f"Select Player {idx+1}",
        df_main["PLAYER"].unique(),
        key=f"player_{idx}"
    )
    available_seasons = df_main[df_main["PLAYER"] == selected_player]["Season"].unique()
    selected_season = st.sidebar.selectbox(
        f"Select Season for {selected_player}", available_seasons, key=f"season_{idx}"
    )

    player_row = df_main[
        (df_main["PLAYER"] == selected_player)
        & (df_main["Season"] == selected_season)
    ]
    if not player_row.empty:
        selected_players.append(player_row.iloc[0])

chosen_metrics = st.multiselect("Select Performance Metrics to Compare", available_metrics)

if selected_players and chosen_metrics:
    st.subheader("Interactive Radar Chart")
    radar_chart = generate_radar_chart(df_main, pd.DataFrame(selected_players), chosen_metrics)
    st.plotly_chart(radar_chart, use_container_width=True)
elif not chosen_metrics:
    st.info("Please select at least one performance metric.")
elif not selected_players:
    st.info("Please select at least one player and season.")
