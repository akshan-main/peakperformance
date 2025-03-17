import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Data pre-processing
@st.cache_data
def load_data():
    df = pd.read_csv('../../dataset/Ratings Combined/playerratingssalaries_100mins.csv')
    df = df.drop(columns=['COUNTRY', 'POS.', 'AGE'], errors='ignore')
    df = df[df['GROSS P/Y (EUR)'] > 0]
    df = df.dropna(subset=['Rating', 'GROSS P/Y (EUR)'])
    return df

# Filter Data
def filter_data(df, selected_season, selected_leagues, selected_positions, selected_clubs):
    if selected_season != 'All':
        df = df[df['Season'] == selected_season]
    if selected_leagues:
        df = df[df['League'].isin(selected_leagues)]
    if selected_positions:
        df = df[df['pos'].isin(selected_positions)]
    if selected_clubs:
        df = df[df['CLUB'].isin(selected_clubs)]
    return df

# Scatter Plot
def plot_scatter_chart(df_filtered, color_col):
    return alt.Chart(df_filtered).mark_circle(size=60, opacity=0.6).encode(
        x=alt.X('Rating:Q', title='Player Rating', scale=alt.Scale(zero=False)),
        y=alt.Y('GROSS P/Y (EUR):Q', title='Gross Yearly Salary (EUR)', scale=alt.Scale(zero=False)),
        color=alt.Color(f'{color_col}:N', title=color_col),
        tooltip=['PLAYER', 'CLUB', 'League', 'Season', 'pos', 'Rating', 'GROSS P/Y (EUR)']
    ).interactive()

# Bar Chart
def plot_bar_chart(df_filtered):
    group_col = 'League' if df_filtered['League'].nunique() > 1 else 'CLUB'
    avg_salary = df_filtered.groupby(group_col, as_index=False)['GROSS P/Y (EUR)'].mean().sort_values('GROSS P/Y (EUR)', ascending=False)
    return alt.Chart(avg_salary).mark_bar().encode(
        x=alt.X('GROSS P/Y (EUR):Q', title='Avg Gross Yearly Salary (EUR)'),
        y=alt.Y(group_col + ':N', sort='-x', title=None),
        color=alt.Color(group_col + ':N', legend=None),
        tooltip=[group_col, 'GROSS P/Y (EUR)']
    )

# Train Salary Prediction Model
@st.cache_data
def train_model(df):
    X = df[['League', 'pos', 'Rating', 'age']]
    y = np.log(df['GROSS P/Y (EUR)'])  # Log-transform salary
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    poly_transformer = PolynomialFeatures(degree=2, include_bias=False)
    transformer = make_column_transformer(
        (OneHotEncoder(drop='first'), ['League', 'pos']),
        remainder='passthrough'
    )
    
    model = make_pipeline(transformer, poly_transformer, LinearRegression())
    model.fit(X_train, y_train)
    max_rating = df.groupby('League')['Rating'].max().to_dict()
    return model, max_rating

# Predict Salary
def predict_salary(model, league, position, rating, age, max_rating_by_league):
    adj_rating = min(rating, max_rating_by_league.get(league, rating))
    pred_log_salary = model.predict(pd.DataFrame([{
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
st.write("Explore player ratings, salaries, and trends across seasons and leagues. Adjust the filters on the sidebar to focus on specific players, teams, or leagues.")

st.subheader("🔍 Performance vs. Salary")
color_option = st.sidebar.radio("Color scatter by", ['League', 'pos'], index=0)
st.altair_chart(plot_scatter_chart(df_filtered, color_option), use_container_width=True)
st.caption("Each point represents a player. Hover to see player details. Zoom in for a closer look.")

st.subheader("💰 Average Salary Trends")
st.altair_chart(plot_bar_chart(df_filtered), use_container_width=True)

model, max_rating_by_league = train_model(df)

st.subheader("📊 Salary Estimator")
st.write("Predict a player's estimated yearly salary based on performance and league trends.")

col1, col2 = st.columns(2)
with col1:
    league_input = st.selectbox("League", sorted(df['League'].unique()), index=0)
    position_input = st.selectbox("Position", ['FW', 'MF', 'DF', 'GK'], index=0)
    rating_input = st.slider("Player Rating", min_value=float(df['Rating'].min()), max_value=9.0, step=0.1, value=7.0)
    age_input = st.slider("Player Age", min_value=int(df['age'].min()), max_value=int(df['age'].max()), value=25)

with col2:
    pred_salary = predict_salary(model, league_input, position_input, rating_input, age_input, max_rating_by_league)
    st.metric(label="💵 Estimated Gross Yearly Salary (EUR)", value=f"€{pred_salary:,.0f}")

st.write("*Salary predictions are based on historical data trends and may not reflect actual market values.*")
