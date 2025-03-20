# PeakPerformance · AI-Driven Football Analytics Project

<p align="center">
    <a href="https://github.com/akshan-main/peakperformance/pulls">
        <img src="https://img.shields.io/github/issues-pr/akshan-main/peakperformance.svg?style=for-the-badge&logo=opencollective" alt="GitHub pull-requests">
    </a>
    <a href="https://github.com/akshan-main/peakperformance/graphs/contributors">
        <img src="https://img.shields.io/github/contributors/akshan-main/peakperformance.svg?style=for-the-badge&logo=bandsintown" alt="GitHub contributors">
    </a>
    <a href="https://github.com/akshan-main/peakperformance/blob/master/LICENSE">
        <img src="https://img.shields.io/github/license/akshan-main/peakperformance?style=for-the-badge&logo=appveyor" alt="GitHub license">
    </a>
    <br>
    <a href="https://github.com/akshan-main/peakperformance">
        <img src="https://img.shields.io/github/repo-size/akshan-main/peakperformance?style=for-the-badge&logo=git" alt="GitHub repository size">
    </a>
    <a href="https://github.com/akshan-main/peakperformance/blob/main/CODE_OF_CONDUCT.md">
        <img src="https://img.shields.io/badge/code%20of-conduct-ff69b4.svg?style=for-the-badge&logo=crowdsource" alt="Code of Conduct">
    </a>
    <br>
    <a href="https://coveralls.io/github/akshan-main/peakperformance?branch=main" > 
        <img src="https://coveralls.io/repos/github/akshan-main/peakperformance/badge.svg?branch=main" alt="Code Coverage"> 
    </a>
    <a href="https://github.com/akshan-main/peakperformance/actions/workflows/python-app.yml">
        <img src="https://github.com/akshan-main/peakperformance/actions/workflows/python-app.yml/badge.svg?branch=main" alt="Build Status">
    </a>
</p>

## ⚽️ Introduction  

What is the average salary a forward at Manchester United earns? Who is the most similar player to 2017 Neymar in the 2023 Premier League season?  We didn't know the answers to these questions either until we built **PeakPerformance**.

**PeakPerformance** is an AI-powered football analytics platform designed to <span style="color:blue">predict seasonal performance</span>, <span style="color:orange">predict salaries</span>,<span style="color:green"> suggest similar players to a particular player</span>,<span style="color:red">simulate player contract negotiations</span>, using machine learning and reinforcement learning. 

It features an interactive Streamlit dashboard with five fully-functional pages.
Explore the deployed web app here: [peakperformance](https://peakperformance.streamlit.app/)

---

## ☰ Table of Contents

- [Introduction](#️introduction)
- [Installation & Setup](#installation--setup)
- [The Team (Contributors)](#the-team-contributors)
- [Features & Pages Overview](#features--pages-overview)
  - [EDA & Salary Analysis](#eda--salary-analysis)
  - [Player Performance Radar](#player-performance-radar)
  - [Player Similarity Matching](#player-similarity-matching)
  - [Player Profile Lookup](#player-profile-lookup)
  - [Contract Negotiator](#contract-negotiator)
- [Dataset](#dataset)
- [Acknowledgments & References](#acknowledgments--references)
- [Show Your Support!](#show-your-support)

---

## Repository Structure
```plaintext
└── dataset
    └── Ratings Combined
        └── filtered_playerratingssalaries.csv
        └── player_data_with_predictions.csv
        └── thousandminsormore.csv
└── docs
    └── in_class_user_stories
        └── user_stories.md
    └── technology_review
        └── technology_review.md
    └── design_spec.md
    └── functional_spec.md
    └── Interaction Diagram.png
    └── milestones.md
└── Python Notebooks
└── peakperformance
    └── home.py
    └── pages
        └── eda_salary.py
        └── player_plot.py
        └── player_profile.py
        └── statmatch.py
        └── the_negotiator.py
    └── backend
        └── train_model.py
    └── tests
        └── test_eda_salary.py
        └── test_home.py
        └── test_player_plot.py
        └── test_player_profile.py
        └── test_statmatch.py
        └── test_the_negotiator.py
        └── test_train_model.py
└── environment.yml
└── requirements.txt
└── README.md

```

---

## 🚀 Installation & Setup

### Requirements
- Python 3.11
- Streamlit
- All dependencies in `requirements.txt`

### Setup Instructions
#### To run the PeakPerformance dashboard locally, please follow these steps:

Clone the GitHub repository to your machine using:

```bash
git clone git@github.com:akshan-main/peakperformance.git
```
Enter the project directory:
```bash
cd peakperformance
```
Create a new conda environment:
```bash
conda env create -f environment.yml
```
Activate the environment:
```bash
conda activate myenv
```
Launch the Streamlit app:
```bash
streamlit run peakperformance/home.py
```

---
## 👥 The Team (Contributors)
- [Akshan Krithick](https://github.com/akshan-main)
- [Balaji Boopal](https://github.com/balajiboopal)
- [Joshua Son](https://github.com/Joshuason55)
---
## 🧩 Features & Pages Overview
✨ EDA & Salary Analysis
- Visualize player salaries across leagues, positions, and clubs.
- Interactive scatter plots (Rating vs Salary) and salary distribution charts.
- Understand league wage disparities and club-wise payroll structures.
<br>

🕸️ Player Performance Radar
- Select 1–10 players and compare them across chosen metrics.
- Dynamic radar charts using Plotly.
- Great for role profiling and visual comparisons.
<br>

🎯 Player Similarity Matching
- Ask questions like “Top 10 players similar to 2022 Messi in 2023 Bundesliga under 25.”
- Natural language queries + Query Parsing + Regular Expressions + cosine similarity.
- Focused on scouting, transfer targeting, and replacing players.

🔍 Player Profile Lookup
- Retrieve profile of football player.
- View predicted ratings over time.
- Chart displays career trajectory with rating trends.

💼 Contract Negotiator
- FIFA-style contract simulation with an RL agent (trained via Deep Q-Learning).
- Offer wages, adjust contract length, and receive counter-offers.
- Optimize strategic contract decisions by analyzing match performance, contract details, and market trends
- The RL agent simulates multi-season decision-making, helping clubs determine whether to extend a player’s contract

---
## 📊 Dataset
For the below mentioned datasets, we used player data from 2017/18 season to 2023/24 season.

### Dataset Structure

- filtered_playerratingssalaries.csv: Used to train the reinforcement learning model.
- thousandminsormore.csv: Filtered ratings, salary data for players with more than 1000 minutes played.
- player_data_with_predictions.csv: Player data with upcoming season ratings prediction (found using XGBoost).
### Data Sources
[FBref website](https://fbref.com/en/) - Player performance metrics data
<br>
[Sofascore website](https://www.sofascore.com/) - Player seasonal ratings data
<br>
[Capology website](https://www.capology.com/) - Player yearly and weekly salary data

---

## 📅 Acknowledgments & References

- The aforementioned data sources.

- Statsbomb & community for radar chart inspiration.

- Streamlit, Scikit-learn, PyTorch, Altair, Plotly, Matplotlib documentations.

---

## 🌟 Show Your Support!
Enjoy using PeakPerformance! If you find this project useful or insightful, please give the repository a star on GitHub. For any issues or enhancement requests, open an issue – we’d love to continue improving the tool with your feedback. Join us in harnessing the power of AI and data to reach new heights in football analytics.