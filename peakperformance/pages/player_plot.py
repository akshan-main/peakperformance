"""
Player Performance Visualization Dashboard

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

    - If `filepath` is provided, it loads from that path.
    - If `filepath` is None, it automatically finds the dataset inside the `dataset` folder.

    Returns:
        df (DataFrame): Processed DataFrame with cleaned column names.
        metrics (list): List of available performance metrics.
    """

    # If no filepath is provided, construct the absolute path dynamically
    if filepath is None:
        # Find the root project directory (assumes `pages/` is inside `peakperformance/`)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        print(project_root)
        filepath = os.path.join(
            project_root,
            "..",
            "dataset",
            "Ratings Combined",
            "playerratingssalaries_100mins.csv"
        )
    df_player = pd.read_csv(filepath)
    exclude_cols = [
        'rk', 'PLAYER', 'nation', 'pos', 'CLUB', 'League', 'age', 'born', 'Season',
        'GROSS P/W (EUR)', 'GROSS P/Y (EUR)', 'GROSS P/Y (EURTAX)'
    ]
    player_metrics = [col for col in df_player.columns if col not in exclude_cols]
    rename_map = {
        col: col.replace('%', 'Percent').replace('p 90', ' per 90')
        for col in player_metrics
    }

    df_player.rename(columns=rename_map, inplace=True)
    player_metrics = [rename_map[col] for col in player_metrics]
    for metric in player_metrics:
        df_player[metric] = pd.to_numeric(df_player[metric], errors='coerce')

    return df_player, player_metrics


# Only execute Streamlit when the script runs directly
df, metrics = load_data()

st.sidebar.title("Filters")

num_players = st.sidebar.slider("Number of Player Selections", 1, 10, 2)

selected_data = []
for i in range(num_players):
    player = st.sidebar.selectbox(
        f"Select Player {i+1}",
        df['PLAYER'].unique(),
        key=f"player_{i}"
    )
    seasons = df[df['PLAYER'] == player]['Season'].unique()
    season = st.sidebar.selectbox(f"Select Season for {player}", seasons, key=f"season_{i}")

    row = df[(df['PLAYER'] == player) & (df['Season'] == season)]
    if not row.empty:
        selected_data.append(row.iloc[0])

selected_metrics = st.multiselect("Select Performance Metrics to Compare", metrics)

if selected_data and selected_metrics:
    st.subheader("Interactive Radar Chart")

    data = pd.DataFrame(selected_data)
    labels = selected_metrics
    fig = go.Figure()

    max_values = df[selected_metrics].max().replace(0, 1e-5)

    for _, row in data.iterrows():
        raw_values = row[labels].astype(float)
        scaled_values = (raw_values / max_values) * 100

        values = scaled_values.fillna(0).tolist()
        values += values[:1]
        hover_texts = [
            f"<b>{label}</b><br>"
            f"Raw: {raw:.2f}<br>"
            f"Scaled: {scaled:.2f}%<br>"
            f"Max: {max_values[label]:.2f}<br>"
            f"<b>{row['PLAYER']} {row['Season']}</b>"
            for label, raw, scaled in zip(labels, raw_values, values[:-1])
        ]

        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=labels + [labels[0]],
            fill='toself',
            name=f"{row['PLAYER']} {row['Season']}",
            line={"width": 3},
            mode="markers+lines",
            marker={"size": 8, "opacity": 0.8},
            hovertemplate="<br>".join([
                "%{text}"
            ]),
            text=hover_texts,
        ))

    fig.update_layout(
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
                "tickfont": {"color": "black", "size": 12}
            }
        },
        showlegend=True,
        title={"text": "<b>Player Performance Radar</b>", "font": {"size": 24}},
        font={"size": 12}
    )
    st.plotly_chart(fig, use_container_width=True)

elif not selected_metrics:
    st.info("Please select at least one performance metric.")
elif not selected_data:
    st.info("Please select at least one player and season.")
