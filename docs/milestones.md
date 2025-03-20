## Milestones

Below is a preliminary plan outlining the major milestones for the project along with the associated tasks, listed in priority order.

### Milestone 1: Data Ingestion and Preprocessing
- **Task 1.1:** Scrape raw data from FBref, Sofascore, and Capology.
- **Task 1.2:** Clean and preprocess the raw data (remove errors, handle missing values).
- **Task 1.3:** Merge data from multiple sources into unified CSV files:
  - `cleaned_player_data.csv`
  - `combined_player_data.csv`
- **Task 1.4:** Engineer additional features and produce final datasets:
  - `filtered_playerratingssalaries.csv`
  - `player_data_features.csv`
  - `player_data_with_predictions.csv`
  - `playerratingssalaries_100mins.csv`

### Milestone 2: Exploratory Data Analysis & Visualization
- **Task 2.1:** Develop interactive scatter plots and bar charts to explore relationships (e.g., player rating vs. salary).
- **Task 2.2:** Create dashboards for comparative analysis across leagues, seasons, and positions.
- **Task 2.3:** Refine visualizations based on initial user feedback and performance considerations.

### Milestone 3: Player Profile & Statistics Module
- **Task 3.1:** Design and implement detailed player profile pages with key metrics and visuals.
- **Task 3.2:** Develop visualizations for player rating trends and seasonal statistics.
- **Task 3.3:** Integrate supplementary player details via external APIs (e.g., TheSportsDB).

### Milestone 4: Contract Negotiation Simulator
- **Task 4.1:** Develop a predictive model using XGBoost for performance forecasting.
- **Task 4.2:** Build the contract negotiation simulator (using RL or rule-based approaches) to generate negotiation outcomes.
- **Task 4.3:** Create a FIFA-style negotiation interface that displays results (including counteroffers and final decisions).
- **Task 4.4:** Test and fine-tune negotiation scenarios for realistic outcomes.

### Milestone 5: Integration & Testing
- **Task 5.1:** Integrate all modules into a cohesive Streamlit application.
- **Task 5.2:** Write comprehensive unit and integration tests.
- **Task 5.3:** Perform user acceptance testing and iterate based on feedback.

### Milestone 6: Documentation & Final Presentation
- **Task 6.1:** Consolidate all design and technical documentation (functional spec, design spec, technology review) into the `docs` folder.
- **Task 6.2:** Finalize user manuals and technical guides.
- **Task 6.3:** Develop and rehearse the final project presentation.
