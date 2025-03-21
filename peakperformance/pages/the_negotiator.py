"""
Football Player Profile Page

This module sets up a Streamlit page for displaying football player profiles
and running a contract negotiation simulation using a Reinforcement Learning agent.

Author: Balaji Boopal
Date: March 16, 2025

Global Variables:
    CSV_PATH (Path): Path to the CSV file containing player salary and rating data.
    PLAYER_DATA (pd.DataFrame): DataFrame with player data loaded from CSV.
    MODEL_PATH (Path): Path to the saved RL model.
    STATE_DIM (int): Dimension of the RL model input state.
    ACTION_DIM (int): Number of possible actions in the RL model.
    RL_agent (ReinforcementLearningAgent): The loaded RL model for contract negotiations.

Returns:
    None
"""

import sys
import time
import random
from pathlib import Path

# ----------------------
# Third-party
# ----------------------
import streamlit as st
import torch
from torch import nn

try:
    from peakperformance.backend.train_model import (
        load_player_data,
        ContractNegotiationEnvironment,
        ReinforcementLearningAgent,
    )
except ModuleNotFoundError:
    # Fallback if the above import fails:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from train_model import (
        load_player_data,
        ContractNegotiationEnvironment,
        ReinforcementLearningAgent,
    )

#############################
# GLOBAL CONSTANTS & SETUP
#############################

CSV_PATH: Path = Path.cwd() / "dataset" / "Ratings Combined" / "filtered_playerratingssalaries.csv"
PLAYER_DATA = load_player_data(CSV_PATH)
MODEL_PATH: Path = Path.cwd() / "assets" / "rl_contract_model.pth"

# The RL model's state and action dimensions:
STATE_DIM: int = 5
ACTION_DIM: int = 4

# Instantiate the RL agent and load the trained model
RL_agent = ReinforcementLearningAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM)
RL_agent.load_model(str(MODEL_PATH))  # loads from 'assets/rl_contract_model.pth'

def fix_translucent_bar() -> None:
    """
    Fixes the translucent blue bar by making it dark yellow with black text.
    """
    st.markdown(
        """
        <style>
            .stAlert {
                background-color: #b8860b !important;
                color: black !important;
                font-weight: bold !important;
                border-radius: 10px !important;
            }
            .stAlert div {
                color: black !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

fix_translucent_bar()

def get_player_details(player_name: str) -> dict:
    """
    Fetches details of a selected player from the dataset.
    """
    player_row = PLAYER_DATA[PLAYER_DATA["PLAYER"] == player_name].iloc[-1]
    return {
        "club": player_row["CLUB"],
        "rating": player_row["Rating"],
        "age": player_row["age"],
        "current_wage": player_row["GROSS P/W (EUR)"],
        "position": player_row["pos"],
    }

def generate_agent_offer(age: int, rating: float, current_wage: float) -> float:
    """
    Generates an agent's initial salary offer based on player age and rating.
    """
    if age < 26:
        return current_wage * random.uniform(1.5, 1.8)
    if age <= 30:
        return current_wage * random.uniform(1.3, 1.5)
    decline_factor = random.uniform(0.7, 1) if rating >= 85 else random.uniform(0.7, 1.0)
    return current_wage * decline_factor

def display_player_card(player_name: str, player_info: dict) -> None:
    """
    Displays the FIFA-style player profile card.
    """
    st.markdown(
        f"""
        <div style="background: rgba(0,0,0,0.85); padding: 15px; border-radius: 10px;">
            <h3 style="color:yellow;"> {player_name}</h3>
            <p><b> Club:</b> {player_info['club']}</p>
            <p><b> Rating:</b> {player_info['rating']}  |   Age: {player_info['age']}</p>
            <p><b> Current Wage:</b> €{int(player_info['current_wage']):,} / week</p>
            <p><b> Position:</b> {player_info['position']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def display_newspaper_announcement(
    player_name: str, proposed_wage: float, contract_length: int, club: str, reward: float
) -> None:
    """
    Displays contract results in a newspaper-style sports announcement.
    """
    newspaper_style = """
        <style>
            .newspaper {
                font-family: 'Times New Roman', serif;
                background: #f5f5dc;
                color: black;
                padding: 20px;
                border-radius: 5px;
                box-shadow: 5px 5px 15px rgba(0, 0, 0, 0.2);
                text-align: center;
                max-width: 750px;
                margin: auto;
                border: 3px solid black;
            }
            .newspaper-header {
                display: flex;
                justify-content: center;
                align-items: center;
                font-size: 48px;
                font-weight: bold;
                text-transform: uppercase;
                border-bottom: 3px solid black;
                padding: 15px;
                margin-bottom: 15px;
            }
            .peak {
                background: red;
                color: white;
                padding: 10px 15px;
                display: inline-block;
                border-radius: 5px 0px 0px 5px;
                font-size: 49px;
            }
            .times {
                background: blue;
                color: white;
                padding: 10px 15px;
                display: inline-block;
                border-radius: 0px 5px 5px 0px;
                font-size: 49px;
            }
            .headline {
                font-size: 30px;
                font-weight: bold;
                text-transform: uppercase;
                text-align: center;
                padding-bottom: 10px;
                margin-bottom: 15px;
            }
            .subheadline {
                font-size: 22px;
                font-weight: bold;
                text-align: center;
                margin-bottom: 15px;
            }
            .body-text {
                font-size: 18px;
                line-height: 1.5;
                text-align: justify;
            }
            .quote {
                font-style: italic;
                font-size: 16px;
                margin-top: 10px;
                text-align: center;
            }
        </style>
    """
    if reward > 0:
        headline = "BLOCKBUSTER DEAL FINALIZED!"
        subheadline = f"{player_name} Secures a Lucrative Contract!"
        body_text = (
            f"{player_name} has agreed to a new contract worth €{int(proposed_wage):,} "
            f"per week for {contract_length} years at {club}."
        )
        quote = (
            f"🗣️ {player_name}: 'I'm excited to continue my journey "
            "with {club}! It's a dream come true.'"
        )
        st.balloons()
    else:
        headline = "CONTRACT TALKS COLLAPSE!"
        subheadline = f"{player_name} Walks Away From Negotiations!"
        body_text = (
            f"{player_name} has rejected the offer of €{int(proposed_wage):,} per week. "
            "Sources indicate the player's camp wanted a better deal."
        )
        quote = ("⚽ Player's Agent: 'We had our conditions, ",
                 "and they werent met. We will explore other options.")

    st.markdown(newspaper_style, unsafe_allow_html=True)
    st.markdown(
        f"""
        <div class="newspaper">
            <div class="newspaper-header">
                <span class="peak">PEAKPERFORMANCE</span>
                <span class="times">TIMES</span>
            </div>
            <div class="headline">{headline}</div>
            <div class="subheadline">{subheadline}</div>
            <p class="body-text">{body_text}</p>
            <p class="quote">{quote}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

#############################
# RL Model Setup
#############################

class DQN(nn.Module):
    """
    DQN architecture from the loaded model. 
    (We keep it here for completeness.)
    """
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Feed forward network
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

STATE_DIM: int = 5
ACTION_DIM: int = 4

# If you want to keep the final loaded model in a separate variable, do:
# (We've already done an RL_agent from train_model, but to keep the code consistent, we override.)
DQN_model = DQN(STATE_DIM, ACTION_DIM)
DQN_model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
DQN_model.eval()

#############################
# Background Setup
#############################

def set_background() -> None:
    """Background Image"""
    image_url = (
        "https://assets.goal.com/images/v3/blta810dc2fdffb1bf9/"
        "1bda86cee7f6af13c4bdcda3c6f763b9e1b1052a.jpg?auto=webp&format=pjpg&width=2048&quality=60"
    )
    st.markdown(
        f"""
        <style>
            .stApp {{
                background-image: url("{image_url}");
                background-size: cover;
                background-position: center;
                background-attachment: fixed;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

set_background()

#############################
# MAIN NEGOTIATION LOGIC
#############################

st.sidebar.image(
    "https://e0.365dm.com/18/08/2048x1152/"
    "skysports-transfer-window-graphic_4385641.jpg?20180810091733",
    width=300,
)

PLAYER_NAME = st.sidebar.selectbox("Select a Player:", PLAYER_DATA["PLAYER"].unique())

# Environment in session state
if "env" not in st.session_state:
    st.session_state.env = ContractNegotiationEnvironment(PLAYER_DATA)
ENV = st.session_state.env

# RL agent from train_model
# We'll re-instantiate in case we want the agent directly in this file:
# But you might skip if you want to rely on the global RL_agent from train_model.
if "RL_agent" not in st.session_state:
    st.session_state.RL_agent = ReinforcementLearningAgent(STATE_DIM, ACTION_DIM)
    st.session_state.RL_agent.load_model(str(MODEL_PATH))

if "state" not in st.session_state:
    st.session_state.state = ENV.reset(PLAYER_NAME)

# Fetch and display info
player_details = get_player_details(PLAYER_NAME)
CLUB, RATING, AGE, CURRENT_WAGE, POSITION = (
    player_details["club"],
    player_details["rating"],
    player_details["age"],
    player_details["current_wage"],
    player_details["position"],
)

# If the user picks a new player, re-reset the environment and generate a new agent wage
if "prev_player" not in st.session_state or st.session_state.prev_player != PLAYER_NAME:
    st.session_state.state = ENV.reset(PLAYER_NAME)
    # Generate new agent wage
    st.session_state.agent_wage = generate_agent_offer(AGE, RATING, CURRENT_WAGE)
    st.session_state.prev_player = PLAYER_NAME

agent_wage = st.session_state.agent_wage

st.sidebar.subheader("Your Contract Proposal")

if "proposed_wage" not in st.session_state:
    st.session_state.proposed_wage = CURRENT_WAGE

PROPOSED_WAGE = st.sidebar.number_input(
    "Proposed Wage (€ per week)",
    min_value=5000,
    max_value=5000000,
    step=1000,
    key="proposed_wage",
)

CONTRACT_LENGTH = st.sidebar.slider("Contract Length (Years)", 1, 6, 3)

col1, col2 = st.columns([1, 2])

with col1:
    display_player_card(PLAYER_NAME, player_details)

with col2:
    st.markdown("<h2 style='color:yellow;'>📜 Contract Negotiation</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background: rgba(0,0,0,0.7); padding: 10px;
        font-weight:900; font-size:1.5rem; border-radius: 5px;">
            <b>Agent</b>: "{PLAYER_NAME} expects a minimum
            wage of <b>€{int(agent_wage):,}</b> per week."
        </div>
        """,
        unsafe_allow_html=True,
    )

# Let the RL agent pick the action automatically
if st.sidebar.button("📜 Submit Offer"):
    with st.spinner("Negotiating..."):
        # Just a small progress bar to simulate negotiation steps
        progress_bar = st.progress(0)
        for i in range(5):
            time.sleep(0.3)
            progress_bar.progress((i + 1) * 20)

        # The RL agent picks an action from the current state
        action_index = st.session_state.RL_agent.select_action(st.session_state.state)

        next_state, negotiation_reward, counteroffer, negotiation_log, done = ENV.step(
            action_index, PROPOSED_WAGE, CONTRACT_LENGTH
        )
        st.session_state.state = next_state

        # Simple constraints to demonstrate custom logic
        if AGE > 32 and CONTRACT_LENGTH > 3:
            st.error(
                "Player rejected because the contract length is too long for an older player. "
                "Try a shorter contract."
            )
        elif AGE < 22 and CONTRACT_LENGTH < 3:
            st.error(
                "Player rejected because the contract length is too short for a younger player. "
                "Try a longer contract."
            )
        else:
            if done:
                display_newspaper_announcement(
                    PLAYER_NAME, PROPOSED_WAGE, CONTRACT_LENGTH, CLUB, negotiation_reward
                )
                if counteroffer:
                    st.warning(counteroffer)
                for log in negotiation_log:
                    st.info(log)
            else:
                st.markdown(
                    """
                    <div style="padding: 20px; border-radius: 10px; text-align: center;
                                font-weight: bold; font-size: 22px; background-color: #f39c12;">
                        <h2>💬 Needs Re-Negotiation</h2>
                        <p>The player's agent wants a better deal. 
                        The RL agent will pick another action next time, or adjust
                        your wage/length.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
        # Decay epsilon so the agent uses its policy more over time
        if st.session_state.RL_agent.epsilon > st.session_state.RL_agent.epsilon_min:
            st.session_state.RL_agent.epsilon *= st.session_state.RL_agent.epsilon_decay
