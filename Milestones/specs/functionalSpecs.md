# Football Analytics Platform - Functional Specification

## **User Stories**

### **1️⃣ End Users (Football Analysts, Scouts, Clubs, Fans)**

#### **Who?**

- Football Analysts.
- Club Scouts.
- General Audience (Football Enthusiasts, Fantasy Football Players).

#### **Wants**

- Predict **a player's peak performance age**.
- View **career trajectory graphs**.
- Get **scouting recommendations (Buy, Scout, Ignore)**.
- Compare **players' historical trends**.

#### **Interaction Methods**

- **Web interface** (Streamlit UI).
- **Filters** for club, position, and league.

#### **Needs**

- A **data-driven way** to evaluate player performance.
- **Track potential talent and declining players**.
- Make **informed scouting decisions**.

#### **Skills**

- **No technical skills required**—simple UI interaction.

---

### **2️⃣ Developer Leveraging the API**

#### **Wants**

- Integrate the **Football Analytics API** into their application.
- Fetch **player insights, peak age predictions, and scouting recommendations**.

#### **Interaction Methods**

- **API calls** to retrieve player statistics.
- **Authentication using API keys**.

#### **Needs**

- **API Documentation** for seamless integration.
- **API Key** for authenticated access.

#### **Skills**

- **Basic understanding of API calls**.
- **Technical skills to integrate API** into web applications.

---

### **3️⃣ Data Scientist**

#### **Wants**

- Work with and **tweak the ML & RL models**.
- Fine-tune **scouting decision-making models**.

#### **Interaction Methods**

- **Model training and evaluation**.
- **Tweaking parameters** and reward functions.

#### **Needs**

- **Access to player data** and historical trends.
- **Tools to test model performance and adjust hyperparameters**.

#### **Skills**

- **Expert-level understanding of ML/RL models**.
- **Working with structured football data**.

---

## **Functional Design**

### **1️⃣ End Users**

#### **Explicit Use Cases**

**User can**:

- **Select a player** to analyze.
- **View player career trajectory graphs**.
- **Receive peak age prediction results**.
- **Get scouting recommendations** (Buy, Scout, Ignore).

**User Flow:**

1. **User selects a player, league, and position**.
2. **System retrieves historical stats** for that player.
3. **System analyzes trends and detects peak seasons**.
4. **User views results as graphs and tables**.
5. **User receives scouting recommendation (Buy, Scout, Ignore)**.
6. **User can export insights for later reference**.

#### **Implicit Use Cases**

- Data visualization of player trajectory.
- Backend **model processing for peak-age detection**.
- **Database queries for past scouting insights**.

---

### **2️⃣ Developers**

#### **Explicit Use Cases**

**Developer can**:

- **Send API requests** to fetch player insights.
- **Retrieve JSON responses** with player statistics.
- **Use API to get peak age prediction & scouting recommendations**.

#### **Implicit Use Cases**

- **User has an API key** for authentication.
- **User has access to API documentation** for guidance.

---

### **3️⃣ Data Scientists**

#### **Explicit Use Cases**

**Data Scientist can**:

- **Train and fine-tune models** for better prediction accuracy.
- **Modify reinforcement learning reward functions**.
- **Test different feature importance techniques**.

#### **Implicit Use Cases**

- **Access to ML and RL models** for tuning.
- **Access to player performance data** for training/testing.

---
