# Functional Specification

## 1. Background

The *Peak Performance* project provides interactive insights into football player performance and salary data. In addition to exploring trends via visualizations, the system simulates contract negotiations using a reinforcement learning (RL) model. The goal is to empower football club analysts, scouts, and journalists to make data-informed decisions about player performance, value, and negotiation strategies using a FIFA-style interface.

## 2. User Profile

**Primary Users:**

- **Football Club Analysts & Scouts:**  
  Have domain expertise in football statistics and salary trends. They can interpret interactive plots, rating trends, and contract simulation outputs.
  
- **Sports Journalists:**  
  Use the system to quickly identify interesting trends or player performance stories and generate compelling narratives.
  
- **Programmers / Data Scientists:**  
  May extend or integrate the system. They have strong Python skills and are comfortable with data analysis libraries.

**Domain Knowledge:**

- Familiarity with football performance metrics (e.g., goals, assists, ratings).
- Basic understanding of salary structures and contract negotiations in sports.
- For technical users: experience with Python, Pandas, and machine learning fundamentals.

## 3. Data Sources

Data for the project is collected by scraping and processing information from three primary websites:
- **FBref:** [https://fbref.com/en/](https://fbref.com/en/)
- **Sofascore:** [https://www.sofascore.com/](https://www.sofascore.com/)
- **Capology:** [https://www.capology.com/](https://www.capology.com/)

The scraped raw data was cleaned and processed to create several CSV files, each with a specific role:

- **cleaned_player_data.csv:**  
  Contains raw player statistics after initial cleaning (e.g., removing missing values).

- **combined_player_data.csv:**  
  Merges multiple data sources to provide a comprehensive view of player performance.

- **filtered_playerratingssalaries.csv:**  
  A refined dataset focusing on player ratings and salary figures, used for visualization and analysis.

- **player_data_features.csv:**  
  Contains engineered features derived from raw data to support advanced analysis.

- **player_data_with_predictions.csv:**  
  Augments player data with predicted ratings or salaries computed by machine learning models.

- **playerratingssalaries_100mins.csv:**  
  A subset of data filtered for specific time intervals (e.g., 100 minutes samples) for detailed performance analysis.

Each CSV is structured with rows representing individual players (or player seasons) and columns for attributes such as player name, club, age, rating, salary, and other performance metrics.

## 4. Use Cases

### Use Case 1: Analyze Player Salary Trends

- **Objective:**  
  Enable users to explore and compare player performance and salary data visually.

- **Detailed Interaction Flow:**

  1. **User Input:**  
     - The user selects a league, season, or specific player from the sidebar filters.
  
  2. **Data Retrieval:**  
     - The **Data Manager** loads the relevant CSV (e.g., `filtered_playerratingssalaries.csv`) and applies the filters.
  
  3. **Visualization:**  
     - The **Visualization Manager** generates interactive scatter plots and bar charts showing player ratings versus salary.
  
  4. **User Exploration:**  
     - The user interacts with the plots (e.g., hovering, zooming) to identify trends and outliers.

  **Interaction Diagram:**

  ```mermaid
  sequenceDiagram
      participant U as User
      participant DM as Data Manager
      participant VM as Visualization Manager
      U->>DM: Request filtered player data
      DM-->>U: Returns filtered DataFrame
      U->>VM: Apply filters & request plots
      VM-->>U: Render interactive charts (scatter, bar)
