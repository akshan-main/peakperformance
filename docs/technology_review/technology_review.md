# Technology Review: Selecting Python Libraries for Peak Performance Analysis

## 1. Introduction
### Project Overview
Our project focuses on analyzing and predicting the *peak performance* of football players by examining historical player statistics. The goal is to identify key performance indicators, build predictive models using time-series and tabular data, and visualize the results in an interactive and insightful way.

To achieve this, we need to carefully select Python libraries that support three key features:

- **Interactive Data Visualization:** Enabling users to explore player performance and salary data via dashboards.
- **Performance Prediction:** Forecasting future player ratings using historical data.
- **Contract Negotiation Simulation (Reinforcement Learning):** Simulating negotiation scenarios to determine optimal contract offers.

## 2. Evaluation of Technology Options

### A. Interactive Data Visualization
For creating interactive dashboards, we compared three popular libraries:

1. **Bokeh**  
   - *Author:* Anaconda, Inc.  
   - *Summary:* Provides interactive plots that can be embedded in web pages; requires some JavaScript knowledge for advanced customization.

2. **Dash**  
   - *Author:* Plotly  
   - *Summary:* A Python framework that uses callbacks for interactive web apps; very flexible but can be verbose to set up.

3. **Streamlit**  
   - *Author:* Streamlit, Inc.  
   - *Summary:* Automatically generates a UI from Python scripts with minimal code; built-in widgets and live reloading make it very attractive for rapid prototyping in data science.

| Feature               | Bokeh                         | Dash                             | Streamlit                          |
|-----------------------|-------------------------------|----------------------------------|------------------------------------|
| Ease of Use           | Moderate (requires some JS)   | Moderate (callback-based)        | Very easy (minimal boilerplate)    |
| Interactivity         | Good (manual setup needed)    | Excellent (flexible callbacks)   | Good (auto-generated widgets)      |
| Deployment            | Embeddable in web pages       | Requires Flask-like setup        | Simple: `streamlit run app.py`     |
| Learning Curve        | Medium                        | Medium-High                      | Low                                |
| Community & Support   | Active community              | Large community                  | Rapidly growing                    |

**Final Choice: Streamlit**  
We chose **Streamlit** because it allows us to quickly build and deploy interactive dashboards with minimal code. It integrates seamlessly with our existing data pipeline (Pandas, Altair, Plotly, and matplotlib are used in various modules) and meets our needs for ease of use and rapid prototyping.

---

### B. Performance Prediction
To forecast future player ratings (peak performance), we evaluated tree-based ensemble libraries, which are well-suited for tabular data:

1. **XGBoost**  
   - *Author:* DMLC/XGBoost community  
   - *Summary:* Known for its excellent predictive performance on structured data and robust feature-importance measures.

2. **LightGBM**  
   - *Author:* Microsoft  
   - *Summary:* Optimized for speed and memory efficiency; performs comparably to XGBoost but may be faster on large datasets.

3. **CatBoost**  
   - *Author:* Yandex  
   - *Summary:* Excels at handling categorical features with minimal preprocessing; can yield competitive results without extensive parameter tuning.

| Feature                    | XGBoost                              | LightGBM                           | CatBoost                         |
|----------------------------|---------------------------------------|------------------------------------|----------------------------------|
| Performance                | Excellent predictive accuracy         | Often faster on large datasets      | Competitive accuracy              |
| Handling Categorical Data  | Requires manual encoding              | Requires manual encoding             | Built-in support                  |
| Memory Efficiency          | Moderate to high                      | Optimized for memory usage           | Similar to XGBoost               |
| Community & Documentation  | Extensive and mature                  | Growing rapidly                      | Good, but smaller community      |
| Ease of Use                | Flexible API; many hyperparameters     | Similar API, typically fewer parameters | Straightforward API              |

**Final Choice: XGBoost**  
We selected **XGBoost** for performance prediction due to its proven track record in tabular data tasks, its robust feature-importance metrics, and its wide community support. Its flexibility and high predictive accuracy meet our project’s forecasting requirements.

---

### C. Contract Negotiation Simulation (Reinforcement Learning)
For simulating contract negotiations, we compared three approaches for RL:

1. **Stable-Baselines3**  
   - *Author:* Open-source community  
   - *Summary:* Provides stable, ready-to-use RL algorithms (like DQN, PPO) with minimal code changes.

2. **RLlib**  
   - *Author:* Part of the Ray framework by Anyscale  
   - *Summary:* Designed for scalable, distributed RL training, but adds complexity for smaller projects.

3. **Custom PyTorch RL**  
   - *Author:* In-house development  
   - *Summary:* A custom-built RL solution using PyTorch, offering maximum flexibility and control; requires more development effort but is tailored to our specific negotiation environment.

| Feature                 | Stable-Baselines3                 | RLlib                             | Custom PyTorch RL                |
|-------------------------|-----------------------------------|-----------------------------------|----------------------------------|
| Ease of Use             | High (prebuilt algorithms)        | Medium (requires Ray integration) | Lower (more custom code required)|
| Flexibility             | Limited to provided algorithms    | Good for large-scale tasks         | Maximum flexibility              |
| Scalability             | Best for single-machine setups    | Designed for distributed training  | Depends on implementation        |
| Integration Complexity  | Minimal                           | Moderate                          | Higher (fully customized)        |

**Final Choice: Custom PyTorch RL**  
We implemented a **custom RL model using PyTorch** for contract negotiation. Although frameworks like Stable-Baselines3 offer prebuilt solutions, our custom implementation provides the flexibility we need to tailor the negotiation dynamics precisely to our domain. It also integrates naturally with our existing PyTorch-based code in other modules.

---

## 3. Overall Drawbacks & Areas of Concern

| Feature/Library           | Drawbacks                                                         |
|---------------------------|-------------------------------------------------------------------|
| **Streamlit**             | Limited advanced UI customization; multi-page apps may require extra work.  |
| **XGBoost**               | Can be memory-intensive on large datasets; requires manual categorical encoding.  |
| **Custom PyTorch RL**     | Requires more development time; lacks built-in scalability and debugging tools of prebuilt libraries.  |

These drawbacks are manageable given our project's scope and resources, and the selected libraries offer the best balance of performance, ease of use, and integration.

---

## 4. Demo Plan

Our demo will include the following components:

- **Interactive Dashboard with Streamlit:**  
  - Show how users can filter player data and view interactive visualizations (scatter plots, bar charts) for performance and salary.
  
- **Performance Prediction with XGBoost:**  
  - Demonstrate the predictive model by inputting player features and displaying predicted future ratings.
  
- **Contract Negotiation Simulation with Custom PyTorch RL:**  
  - Run a short simulation where the user enters a wage and contract length, and the custom RL model outputs a negotiation result.
  
The demo will consist of code snippets and live output, illustrating how minimal code is required to achieve interactive functionality and how the models integrate within the Streamlit interface.

---

## 5. Conclusion

This Technology Review guided our evaluation of Python libraries for three critical features of our project:

- **Interactive Data Visualization:** Chosen: **Streamlit**
- **Performance Prediction:** Chosen: **XGBoost**
- **Contract Negotiation Simulation (RL):** Chosen: **Custom PyTorch RL**

These selections ensure that our project is built on robust, efficient, and easy-to-use libraries, allowing us to deliver an interactive, data-driven analysis of peak player performance that meets the needs of both technical and non-technical users.

*Next Steps:*
- Fine-tune our XGBoost model for improved accuracy.
- Optimize our custom RL model to better simulate realistic negotiation scenarios.
- Finalize our Streamlit dashboard and prepare for the live demo presentation.
