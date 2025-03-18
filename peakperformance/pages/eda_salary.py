"""
Football Salary Distribution & Salary Prediction Dashboard

This module loads player rating and salary data, allows filtering, and provides salary predictions
based on historical salary trends using a machine learning model.

Author: Akshan Krithick
Date: March 16, 2025
"""
import os
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

@st.cache_data
def load_data():
    """
    Loads player rating and salary data from a CSV file.

    Returns:
        pd.DataFrame: A DataFrame containing cleaned player data.
    """
    csv_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__), "..", "..", "dataset",
            "Ratings Combined", "thousandminsormore.csv"
        )
    )
    df_output = pd.read_csv(csv_path)
    df_output = df_output.drop(columns=['COUNTRY', 'POS.', 'AGE'], errors='ignore')
    df_output = df_output[df_output['GROSS P/Y (EUR)'] > 0]
    df_output = df_output.dropna(subset=['Rating', 'GROSS P/Y (EUR)'])
    return df_output

def filter_data(df_input, select_season, select_leagues, select_positions, select_clubs):
    """
    Filters the player data based on selected season, leagues, positions, and clubs.

    Args:
        df_input (pd.DataFrame): The original DataFrame containing player data.
        select_season (str): The selected season to filter on.
        select_leagues (list): List of selected leagues.
        select_positions (list): List of selected player positions.
        select_clubs (list): List of selected clubs.

    Returns:
        pd.DataFrame: A filtered DataFrame based on selected criteria.
    """
    if select_season != 'All':
        df_input = df_input[df_input['Season'] == select_season]
    if selected_leagues:
        df_input = df_input[df_input['League'].isin(select_leagues)]
    if selected_positions:
        df_input = df_input[df_input['pos'].isin(select_positions)]
    if selected_clubs:
        df_input = df_input[df_input['CLUB'].isin(select_clubs)]
    return df_input


def plot_scatter_chart(df_input_filtered, color_col):
    """
    Creates an Altair scatter plot for player rating vs. salary.

    Args:
        df_input_filtered (pd.DataFrame): Filtered DataFrame containing player data.
        color_col (str): Column to use for coloring the scatter points.

    Returns:
        alt.Chart: An Altair scatter plot object.
    """
    return alt.Chart(df_input_filtered).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X('Rating:Q', title='Player Rating', scale=alt.Scale(zero=False)),
        y=alt.Y('GROSS P/Y (EUR):Q', title='Gross Yearly Salary (EUR)',
                scale=alt.Scale(zero=False)),
        color=alt.Color(f'{color_col}:N', title=color_col),
        tooltip=['PLAYER', 'CLUB', 'League', 'Season', 'pos', 'Rating', 'GROSS P/Y (EUR)']
    ).interactive()

def plot_bar_chart(df_filtered_bar):
    """
    Creates an Altair bar chart for average salary distribution across leagues or clubs.

    Args:
        df_filtered_bar (pd.DataFrame): Filtered DataFrame containing player data.

    Returns:
        alt.Chart: An Altair bar chart object.
    """
    group_col = 'League' if df_filtered_bar['League'].nunique() > 1 else 'CLUB'
    avg_salary = (
        df_filtered_bar.groupby(group_col, as_index=False)['GROSS P/Y (EUR)']
        .mean()
        .sort_values('GROSS P/Y (EUR)', ascending=False)
    )
    return alt.Chart(avg_salary).mark_bar().encode(
        x=alt.X('GROSS P/Y (EUR):Q', title='Avg Gross Yearly Salary (EUR)'),
        y=alt.Y(group_col + ':N', sort='-x', title=None),
        color=alt.Color(group_col + ':N', legend=None),
        tooltip=[group_col, 'GROSS P/Y (EUR)']
    )

# Train Salary Prediction Model
@st.cache_data
def train_model(df_model):
    """
    Trains a polynomial regression model to predict player salaries.

    Args:
        df_model (pd.DataFrame): The dataset containing player features and salaries.

    Returns:
        tuple: A trained regression model and a dictionary mapping leagues to max ratings.
    """
    x = df_model[['League', 'pos', 'Rating', 'age']]
    y = np.log(df_model['GROSS P/Y (EUR)'])  # Log-transform salary

    x_train, _, y_train, _ = train_test_split(x, y, test_size=0.2, random_state=42)

    poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
    transformer = make_column_transformer(
        (OneHotEncoder(drop='first'), ['League', 'pos']),
        remainder='passthrough'
    )

    train_model_res = make_pipeline(transformer, poly_transformer, LinearRegression())
    train_model_res.fit(x_train, y_train)
    max_rating = df_model.groupby('League')['Rating'].max().to_dict()
    return train_model_res, max_rating

# Predict Salary
def predict_salary(model_input, player_info_input, max_rating_by_league_input):
    """
    Predicts a player's salary using the trained regression model.

    Args:
        model (Pipeline): The trained salary prediction model.
        player_info (dict): A dictionary containing player's league, position, rating, and age.
            Keys: "League" (str), "pos" (str), "Rating" (float), "age" (int)
        max_rating_by_league (dict): Dictionary mapping leagues to their max rating.

    Returns:
        float: Predicted gross yearly salary (EUR).
    """
    league = player_info_input["League"]
    position = player_info_input["pos"]
    rating = player_info_input["Rating"]
    age = player_info_input["age"]

    adj_rating = min(rating, max_rating_by_league_input.get(league, rating))
    pred_log_salary = model_input.predict(pd.DataFrame([{
        "League": league, "pos": position, "Rating": adj_rating, "age": age
    }]))[0]
    return float(np.exp(pred_log_salary))

# Streamlit UI Setup
df = load_data()

st.sidebar.header("Filters")
season_options = ['All'] + sorted(df['Season'].unique().tolist())
selected_season = st.sidebar.selectbox("Season", season_options, index=len(season_options) - 1)
selected_leagues = st.sidebar.multiselect("League", sorted(df['League'].unique()))
selected_positions = st.sidebar.multiselect("Position", sorted(df['pos'].unique()))
selected_clubs = st.sidebar.multiselect("Club (optional)", sorted(df['CLUB'].unique()))

df_filtered = filter_data(df, selected_season, selected_leagues, selected_positions, selected_clubs)

st.title("⚽ Football Player Performance & Salary Explorer")
st.write(
    "Explore player ratings, salaries, and trends across seasons and leagues. "
    "Adjust the filters on the sidebar to focus on specific players, teams, or leagues."
)

st.subheader("🔍 Performance vs. Salary")
color_option = st.sidebar.radio("Color scatter by", ['League', 'pos'], index=0)
st.altair_chart(plot_scatter_chart(df_filtered, color_option), use_container_width=True)
st.caption(
    "Each point represents a player. Hover to see player details. Zoom in for a closer look."
)

st.subheader("💰 Average Salary Trends")
st.altair_chart(plot_bar_chart(df_filtered), use_container_width=True)

model, max_rating_by_league = train_model(df)

st.subheader("📊 Salary Estimator")
st.write("Predict a player's estimated yearly salary based on performance and league trends.")

col1, col2 = st.columns(2)
with col1:
    league_input = st.selectbox("League", sorted(df['League'].unique()), index=0)
    position_input = st.selectbox("Position", ['FW', 'MF', 'DF', 'GK'], index=0)
    rating_input = st.slider(
        "Player Rating",
        min_value=float(df['Rating'].min()),
        max_value=9.0,
        step=0.1,
        value=7.0
    )
    age_input = st.slider(
        "Player Age",
        min_value=int(df['age'].min()),
        max_value=int(df['age'].max()),
        value=25
    )

with col2:
    player_info = {
        "League": league_input,
        "pos": position_input,
        "Rating": rating_input,
        "age": age_input
    }
    pred_salary = predict_salary(model, player_info, max_rating_by_league)
    st.metric(label="💵 Estimated Gross Yearly Salary (EUR)", value=f"€{pred_salary:,.0f}")

st.write(
    "*Salary predictions are based on historical data trends and may not reflect "
    "actual market values.*"
)
