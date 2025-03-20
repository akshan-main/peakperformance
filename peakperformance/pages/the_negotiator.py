"""
Contract Simulator: FIFA-Style Player Contract Negotiation using RL.

This module simulates contract negotiations for football players using a reinforcement
learning model. It loads player data, initializes an RL model, and sets up a Streamlit
interface to display player information, contract negotiation details, and results in a
FIFA-style layout.

Global Variables:
    CSV_PATH (Path): Path to the CSV file containing player salary and rating data.
    PLAYER_DATA (pd.DataFrame): DataFrame with player data loaded from CSV.
    MODEL_PATH (Path): Path to the saved RL model.
    STATE_DIM (int): Dimension of the RL model input state.
    ACTION_DIM (int): Number of possible actions in the RL model.
    MODEL (DQN): The loaded RL model for contract negotiations.

Returns:
    None
"""
import sys
import time
import random
from pathlib import Path
import streamlit as st
import torch
from torch import nn

try:
    from peakperformance.backend.train_model import load_player_data, ContractNegotiationEnvironment
except ModuleNotFoundError:
    sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
    from peakperformance.backend.train_model import load_player_data, ContractNegotiationEnvironment



CSV_PATH: Path = Path.cwd() / "dataset" / "Ratings Combined" / "filtered_playerratingssalaries.csv"

PLAYER_DATA = load_player_data(CSV_PATH)
MODEL_PATH: Path = Path.cwd() / "assets" / "rl_contract_model.pth"

def fix_translucent_bar() -> None:
    """
    Fixes the translucent blue bar by making it dark yellow with black text.

    Returns:
        None
    """
    st.markdown(
        """
        <style>
            /* Fix translucent messages like st.info() */
            .stAlert {
                background-color: #b8860b !important;
                color: black !important;
                font-weight: bold !important;
                border-radius: 10px !important;
            }
            /* Fix text color inside alert */
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

    Parameters:
        player_name (str): The name of the player.

    Returns:
        dict: A dictionary containing the club, rating, age, current wage, and position.
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

    Parameters:
        age (int): The player's age.
        rating (float): The player's rating.
        current_wage (float): The player's current wage.

    Returns:
        float: The generated agent offer.
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

    Parameters:
        player_name (str): The name of the player.
        player_info (dict): A dictionary containing player details.

    Returns:
        None
    """
    st.markdown(
        f"""
        <div style="background: rgba(0,0,0,0.85); padding: 15px; border-radius: 10px;">
            <h3 style="color:yellow;">📌 {player_name}</h3>
            <p><b>🏟️ Club:</b> {player_info['club']}</p>
            <p><b>🎯 Rating:</b> {player_info['rating']}  |  📅 Age: {player_info['age']}</p>
            <p><b>💰 Current Wage:</b> €{int(player_info['current_wage']):,} / week</p>
            <p><b>🛡️ Position:</b> {player_info['position']}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_newspaper_announcement(
    player_name: str, proposed_wage: float, contract_length: int, club: str, reward: float
) -> None:
    """
    Displays contract results in a newspaper-style sports announcement.

    Parameters:
        player_name (str): The name of the player.
        proposed_wage (float): The proposed wage.
        contract_length (int): The contract length in years.
        club (str): The player's club.
        reward (float): The reward from the negotiation.

    Returns:
        None
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
            f"🗣️ {player_name}: 'I'm excited to continue my journey with {club}! "
            "It's a dream come true.'"
        )
        st.balloons()
    else:
        headline = "CONTRACT TALKS COLLAPSE!"
        subheadline = f"{player_name} Walks Away From Negotiations!"
        body_text = (
            f"{player_name} has rejected the offer of €{int(proposed_wage):,} per week. "
            "Sources indicate the player's camp wanted a better deal."
        )
        quote = (
            "⚽ Player's Agent: 'We had our conditions, and they weren’t met. "
            "We will explore other options.'"
        )

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


class DQN(nn.Module):
    """
    Deep Q-Network Model for contract negotiations.

    Parameters:
        state_dim (int): Dimension of the input state.
        action_dim (int): Number of possible actions.
    """
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Parameters:
            x (torch.Tensor): Input tensor representing the state.

        Returns:
            torch.Tensor: Output tensor with Q-values for each action.
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Initialize Model
STATE_DIM: int = 5
ACTION_DIM: int = 4
MODEL = DQN(STATE_DIM, ACTION_DIM)
MODEL.load_state_dict(torch.load(MODEL_PATH))
MODEL.eval()


def set_background() -> None:
    """
    Sets a full-page background image in Streamlit.

    Returns:
        None
    """
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


# Sidebar Player Selection
st.sidebar.image(
    "https://e0.365dm.com/18/08/2048x1152/"
    "skysports-transfer-window-graphic_4385641.jpg?20180810091733",
    width=300,
)
PLAYER_NAME: str = st.sidebar.selectbox("Select a Player:", PLAYER_DATA["PLAYER"].unique())

# Initialize environment AFTER player selection
if "env" not in st.session_state:
    st.session_state.env = ContractNegotiationEnvironment(PLAYER_DATA)
ENV = st.session_state.env

if "state" not in st.session_state:
    st.session_state.state = ENV.reset(PLAYER_NAME)

# Rename the global player_info to player_details to avoid shadowing warnings.
player_details = get_player_details(PLAYER_NAME)
CLUB, RATING, AGE, CURRENT_WAGE, POSITION = (
    player_details["club"],
    player_details["rating"],
    player_details["age"],
    player_details["current_wage"],
    player_details["position"],
)

agent_wage: float = generate_agent_offer(AGE, RATING, CURRENT_WAGE)

st.sidebar.subheader("💼 Your Contract Proposal")

# Ensure session state has a default wage
if "proposed_wage" not in st.session_state:
    st.session_state.proposed_wage = CURRENT_WAGE

# Number input with proper session state binding
PROPOSED_WAGE: float = st.sidebar.number_input(
    "Proposed Wage (€ per week)",
    min_value=5000,
    max_value=5000000,
    step=1000,
    key="proposed_wage",
)

# Contract length selection
CONTRACT_LENGTH: int = st.sidebar.slider("Contract Length (Years)", 1, 6, 3)

# FIFA-Style Player Card Layout
col1, col2 = st.columns([1, 2])
with col1:
    display_player_card(PLAYER_NAME, player_details)

with col2:
    st.markdown("<h2 style='color:yellow;'>📜 Contract Negotiation</h2>", unsafe_allow_html=True)
    st.markdown(
        f"""
        <div style="background: rgba(0,0,0,0.7); padding: 10px; border-radius: 5px;">
            <b>Agent</b>: "{PLAYER_NAME} expects a wage of <b>€{int(agent_wage):,}</b> per week."
        </div>
        """,
        unsafe_allow_html=True,
    )

if st.sidebar.button("📜 Submit Offer"):
    progress_bar = st.progress(0)
    for i in range(5):
        time.sleep(0.3)
        progress_bar.progress((i + 1) * 20)

    next_state, negotiation_reward, counteroffer, negotiation_log, done = ENV.step(
        0, PROPOSED_WAGE, CONTRACT_LENGTH
    )

    DECISION_TEXT = ""
    FEEDBACK_TEXT = ""
    COLOR_CLASS = ""

    if done:
        display_newspaper_announcement(PLAYER_NAME, PROPOSED_WAGE, CONTRACT_LENGTH,
                                       CLUB, negotiation_reward)
        if counteroffer:
            st.warning(counteroffer)
        for log in negotiation_log:
            st.info(log)
    elif negotiation_reward == 0:  # Ensures it's not accepted or rejected
        st.markdown(
            """
            <div style="padding: 20px; border-radius: 10px; text-align: center; font-weight: bold;
                        font-size: 22px; background-color: #f39c12;">
                <h2>💬 Needs Re-Negotiation</h2>
                <p>Player's agent wants a better deal.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # Display Feedback
    st.markdown(
        f"""
        <div style="padding: 20px; border-radius: 10px; text-align: center; font-weight: bold;
                    font-size: 22px;">
            <h2>{DECISION_TEXT}</h2>
            <p>{FEEDBACK_TEXT}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Update session state and trigger rerun
    st.session_state.state = next_state
