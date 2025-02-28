import pandas as pd

df = pd.read_csv("player_data.csv")

# Sample ML Model (Dummy Prediction for Now)
def ml_model_predict(player_name):
    return 85.0  # Replace with actual ML prediction

def get_player(player_name: str):
    player = df[df["player"] == player_name].iloc[0]
    # Fetch images
    player_image = f"https://storage.example.com/players/{player_name}.png"
    club_logo = f"https://storage.example.com/clubs/{player['club']}.png"
    national_flag = f"https://storage.example.com/flags/{player['nationality']}.png"

    # Get ML prediction
    predicted_rating = ml_model_predict(player_name)

    return {
        "name": player["player"],
        "club": player["club"],
        "position": player["position"],
        "nationality": player["nationality"],
        "ml_prediction": predicted_rating,
        "player_image": player_image,
        "club_logo": club_logo,
        "national_flag": national_flag
    }
