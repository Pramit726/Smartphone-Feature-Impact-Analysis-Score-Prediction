# Technical Documentation: Smartphone Specification Analysis

## 1. Requirements Gathering

### A. Scope Clarification

1. What is the system supposed to do?
   - Predict Smartprix specification scores for smartphones based on their features.
   - Provide interpretability to help stakeholders understand which features have the most significant impact on the score.
   - Enable stakeholders to input features for a hypothetical smartphone and predict its specification score.
2. Who are the primary users?
   - Product managers of smartphone brands aiming to improve product design and strategy.
3. What are the boundaries of the system?
   - The system focuses only on predicting and interpreting specification scores, not on direct sales or customer feedback analysis.
   - It relies on structured smartphone specification data (e.g., processor, camera, battery) and existing Smartprix ratings.
4. What problem are we solving for the stakeholders?
   - Helping them design smartphones with higher specification scores by understanding feature impact.
   - Enabling informed decisions during product development with reliable predictions and actionable insights.

### B. Functional Requirements

1. Prediction Functionality:
   - Predict the Smartprix specification score based on the provided features using a regression model.
2. Interpretability Features:
   - Provide feature importance rankings (e.g., which features have the highest impact on scores).
   - Offer partial dependency plots or SHAP-based interpretations for deeper insights.
3. Hypothetical Device Scoring:
   - Allow users to input custom feature values for a new smartphone and generate its predicted score.
4. Data Handling:
   - Ingest, preprocess, and validate smartphone specification data.
   - Handle missing values, normalize numerical features, and encode categorical ones.
5. Web Application Integration:
   - Host a user-friendly interface for stakeholders to input features, view predictions, and explore interpretability results.
6. Model Evaluation and Continuous Learning:
   - Evaluate model performance with metrics like R² and MAE.

### C. Non-functional Requirements

1. Scalability:
   - Ensure the system can handle large datasets with hundreds of features and thousands of data points.
2. Performance:
   - Predictions should be generated in under 2 seconds to ensure seamless interaction in the web application.
3. Reliability:
   - The system should provide consistent results across multiple runs, with minimal variance.
4. Interpretability:
   - The outputs must be easily understandable by non-technical stakeholders, with visual aids like bar charts or plots.
5. Maintainability:
   - Use modular and well-documented code to enable easy updates and debugging.
6. Usability:
   - Provide an intuitive interface for non-technical users, focusing on simplicity and clarity.

## 2. Architecture Planning

### High-Level Architecture

1. Data Collection Layer:

   - **Purpose**: Collect data on smartphone specifications and corresponding Smartprix scores.
   - **Components**:
     - Data sources: APIs, databases, or web scraping tools to collect data from platforms like Smartprix or manufacturer websites.
     - Pre-processing pipeline: A script or framework to clean, transform, and validate raw data.
   - **Why**: Ensures the input data is accurate, consistent, and ready for modeling. Reliable data forms the foundation of a trustworthy system.

2. Data Storage Layer:

   - **Purpose**: Store pre-processed smartphone specification data and Smartprix scores.
   - **Components**:
     - Technology: PostgreSQL for structured data storage.
   - **Why PostgreSQL**:
     - PostgreSQL is ideal for storing tabular data such as smartphone features and specification scores.
     - Offers powerful querying capabilities and ensures data consistency, essential for reliable predictions.

3. Modeling and Analytics Layer:

   - **Purpose**: Train regression models to predict specification scores and provide interpretability.
   - **Components**:
     - ML models: Techniques like Linear Regression, Random Forests, or XGBoost for accurate and interpretable predictions.
     - Interpretability tools: SHAP (SHapley Additive exPlanations) for feature importance analysis.
   - **Why**:
     - Regression models provide accurate predictions for continuous outputs like specification scores.
     - SHAP enhances stakeholder trust by explaining why specific features drive certain scores.

4. Application Layer:

   - **Purpose**: Provide a user-friendly interface for stakeholders to interact with the model.
   - **Components**:
     - Technology: Streamlit for the web application.
   - **Why Streamlit**:
     - Rapid prototyping and deployment of interactive dashboards.
     - Minimal development effort to create visually appealing interfaces for stakeholders.
     - Built-in widgets for input forms, graphs, and tables to simplify user interaction.

5. Deployment Layer:

   - **Purpose**: Host the application and make it accessible to stakeholders.
   - **Components**:
     - Technology: Streamlit Cloud for hosting the web application.
   - **Why Streamlit Cloud**:
     - Streamlined and cost-effective solution for hosting proof-of-concept (PoC) projects.
     - Eliminates infrastructure setup overhead, allowing focus on app functionality.
     - Optimized for Streamlit apps, ensuring seamless deployment and easy sharing of results with stakeholders.

### Technology Choices

| **Layer**              | **Technology**        | **Why Chosen**                                                                      |
| ---------------------- | --------------------- | ----------------------------------------------------------------------------------- |
| Data Preprocessing     | Pandas, Scikit-learn  | Comprehensive tools for data cleaning, feature engineering, and transformation.     |
| Modeling Frameworks    | Scikit-learn, XGBoost | Easy-to-use for regression tasks; XGBoost excels at capturing complex interactions. |
| Interpretability Tools | SHAP                  | Provides actionable insights into feature importance.                               |
| Database Choice        | PostgreSQL            | Strong relational database with excellent querying capabilities.                    |
| Web Application        | Streamlit             | Simplifies building dashboards with minimal development effort.                     |
| Deployment Platform    | Streamlit Cloud       | Cost-effective and optimized for Streamlit apps.                                    |

### Evaluation Strategy and Key Metrics

| **Layer**              | **Evaluation Strategy**                                                                             | **Key Metrics**                              |
| ---------------------- | --------------------------------------------------------------------------------------------------- | -------------------------------------------- |
| Modeling Frameworks    | Cross-validation, parameter tuning (grid search), and feature importance analysis.                  | R² (>0.8), MAE (±5 points).                  |
| Interpretability Tools | Visual inspection of SHAP plots, stakeholder feedback on interpretability.                          | SHAP value consistency, usability.           |
| Web Application        | Usability tests with stakeholders, app responsiveness checks.                                       | Interaction time (<2 seconds).               |
| Deployment Platform    | Deployment time tracking, uptime monitoring.                                                        | Uptime percentage (>99%).                    |

### Key Metrics

| **Metric**                    | **Technical Meaning**                                                | **Business Relevance**                                                  | **Example Outcome**                                                |
| ----------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------------------------- | ------------------------------------------------------------------ |
| **R-Squared (R²)**            | Explains how much variance in scores is captured by the model.       | High R² (> 0.8) ensures reliability for strategic planning.             | "85% of score variance explained by smartphone features."          |
| **Mean Absolute Error (MAE)** | Quantifies average deviation of predicted scores from actual scores. | Low MAE (±5 points) boosts trust in predictions for critical decisions. | "Predicted flagship score of 90 typically within 85-95."           |
| **Feature Consistency**       | Measures interpretability tool consistency in feature ranking.       | Ensures actionable insights for design prioritization.                  | "Camera quality consistently ranks as top feature across devices." |
| **Interaction Time**          | Measures web app responsiveness.                                     | Enhances user satisfaction with seamless interactions.                  | "Predictions generated within 1.5 seconds for 95% of inputs."      |

## 3. Data Pipeline Design

### A. Data Collection

#### Overview
- Scraping data from **Smartprix** website using **Selenium** (for dynamic content) and **BeautifulSoup** (for parsing HTML).
- Data includes smartphone specifications for predictive modeling.

#### Data Collection Methodology
1. **Selenium**: 
   - Automates browser actions: applies filters, clicks, and scrolls to load dynamic content.
   - Saves page source to a local HTML file after interacting with the website.
2. **BeautifulSoup**:
   - Parses the saved HTML file.
   - Extracts smartphone details such as model, price, specs, and ratings.

#### Key Data Fields
- **Brand Name**: Smartphone brand (e.g., Apple, Samsung).
- **Model**: Specific model name.
- **Price**: Retail price.
- **Rating**: Specification score.
- **Specs**: processor, RAM, battery, display, camera, expandable storage, OS.

For a detailed data dictionary, refer to the [Data_Dictionary](../references/Mobile_Insight_Data_Dictionary.xlsx).

#### Data Scraping Process
- **Initialization**: Set up Selenium with `chromedriver.exe`.
- **Opening Page**: Load the target Smartprix URL.
- **Applying Filters**: Use XPath to apply filters (e.g., smartphone category).
- **Scrolling**: Simulate scrolling to load all products.
- **Save HTML**: Store page source in a local HTML file for extraction.
- **Close Driver**: Quit the Selenium driver after scraping.

#### Data Export
- **Extraction**: Parse HTML to extract required data.
- **Storage**: Convert data to a pandas DataFrame.
- **Export**: Save data as a CSV file for further analysis.

#### Tools Used
- **Selenium**: Automates web interactions.
- **BeautifulSoup**: Parses HTML for data extraction.
- **Pandas**: Stores and exports data.

## Deployment and Serving

### Overview

The goal of this section is to outline the deployment process and the serving architecture for making the machine learning model accessible to stakeholders via a web application. This section details how the trained predictive model will be deployed, how real-time predictions will be provided, and how interpretability features will be integrated for feature analysis.

### Components

1. **Web Application Hosting**:
   - **Platform**: Streamlit Cloud
   - **Description**: The web application will be deployed on Streamlit Cloud to provide an interactive and user-friendly interface for stakeholders. Streamlit Cloud offers a simple and efficient deployment solution, ideal for proof-of-concept (PoC) projects, with minimal infrastructure overhead.
   - **Functionality**: The web application will allow users to input hypothetical smartphone features and receive predicted Smartprix specification scores in real-time. Additionally, users will be able to view feature importance via SHAP (SHapley Additive exPlanations) visualizations to understand the contribution of each feature to the predicted score.
   - **Reason for Selection**: Streamlit Cloud is ideal for PoC projects due to its simplicity, rapid deployment, and minimal configuration requirements. It supports Python-based machine learning models and provides built-in widgets for interactive user inputs.

2. **Model Deployment**:
   - **Serialization Format**: Pickle or Joblib
   - **Description**: The trained predictive model will be serialized using Pickle or Joblib to facilitate its deployment. These formats allow the model to be stored efficiently and loaded quickly during inference.
   - **Functionality**: Once serialized, the model will be loaded by the Streamlit application for real-time predictions. The model takes smartphone feature inputs, processes them, and outputs the predicted Smartprix specification score.
   - **Reason for Selection**: Pickle and Joblib are widely used in Python-based machine learning workflows for model serialization. They support fast loading and make the deployment process seamless.

3. **Feature Interpretability**:
   - **Library**: SHAP (SHapley Additive exPlanations)
   - **Description**: SHAP will be integrated into the application to provide interpretability of the model's predictions. It will generate plots that show the contribution of each feature to the predicted specification score, helping stakeholders understand the impact of different features on the model’s output.
   - **Functionality**: SHAP values will be computed for each prediction, and corresponding plots will be displayed on the web application. This helps stakeholders make data-driven decisions about feature prioritization for future smartphone designs.
   - **Reason for Selection**: SHAP is a widely used library for model interpretability, providing reliable and easily understandable visual explanations of machine learning model predictions.

4. **Real-Time Predictions**:
   - **Platform**: Streamlit Widgets and Python Backend
   - **Description**: The web application will allow stakeholders to input smartphone feature values via interactive widgets, such as sliders and text input fields. Once the features are provided, the model will predict the corresponding Smartprix specification score.
   - **Functionality**: The application will utilize Streamlit's widget framework to capture user inputs, pass them to the model for inference, and display the predicted score in real time.
   - **Reason for Selection**: Streamlit provides an efficient framework for building interactive dashboards with minimal code, making it easy to implement real-time model predictions.

5. **Monitoring and Observability**:
   - **Stack**: Prometheus-Grafana Stack
   - **Description**: Prometheus and Grafana will be used to monitor the application’s performance, resource usage, and reliability. Prometheus will collect metrics, and Grafana will visualize them through interactive dashboards.
   - **Functionality**: The monitoring stack will track key metrics such as response time, user interactions, and application uptime. Alerts will be configured for any anomalies or performance bottlenecks.
   - **Reason for Selection**: Prometheus and Grafana are industry-standard tools for monitoring and observability, offering a robust solution for tracking application performance and ensuring high availability.

### Deployment Process

1. **Prepare the Environment**:
   - **Streamlit Setup**: Create a Streamlit Cloud account and initialize the project repository on GitHub.
   - **Dependencies**: List all necessary Python libraries in a `requirements.txt` file, including libraries for machine learning (e.g., `scikit-learn`, `xgboost`), data processing (e.g., `pandas`), and visualization (e.g., `shap`, `matplotlib`, `plotly`).
   - **Model Serialization**: Serialize the trained model using Pickle or Joblib and save it to the project repository.

2. **Code Deployment**:
   - **Repository Hosting**: Host the project code and serialized model in a GitHub repository. This repository will contain the Streamlit app script, model file, and any auxiliary scripts required for data processing and model inference.
   - **Streamlit Deployment**: Link the GitHub repository to Streamlit Cloud. Once connected, Streamlit Cloud will automatically deploy the application. Any updates pushed to the GitHub repository will trigger an automatic redeployment on Streamlit Cloud.

3. **Web Application Interface**:
   - **Input Widgets**: Use Streamlit's widgets (e.g., sliders, text boxes) to create an intuitive interface where users can input hypothetical smartphone features such as processor type, camera quality, and battery life.
   - **Prediction Output**: After the user provides input, the model will generate a predicted Smartprix specification score, which will be displayed on the web application.
   - **Feature Impact Visualization**: SHAP plots will be integrated into the interface to visually explain how each feature influences the predicted score.

4. **Model Inference**:
   - **Process**: When a user provides input, the Streamlit app will:
     - Parse the input values and convert them into a format suitable for the model.
     - Load the serialized model from the Pickle/Joblib file.
     - Use the model to predict the Smartprix specification score based on the provided features.
     - Display the predicted score and the corresponding SHAP value plot showing feature importance.

5. **Monitoring and Maintenance**:
   - **Future Monitoring**: Prometheus-Grafana stack will be used to monitor the application’s performance, uptime, and user interactions to ensure reliability and scalability in a production environment.
   - **Model Updates**: If new data becomes available or if the model needs retraining, updates will be made to the model and the Streamlit application will be redeployed with the updated model.

### Scalability Considerations

- **For PoC**: The application is expected to handle a limited number of users, primarily internal stakeholders (product managers, analysts).
- **For Future Expansion**: Should the project move to full-scale production, considerations for scaling the application will include moving to more robust cloud infrastructure (e.g., AWS, Google Cloud), optimizing model inference time, and potentially implementing load balancing for increased user traffic.

### Summary of Deployment Architecture

| **Component**             | **Technology**               | **Purpose**                                           |
|---------------------------|------------------------------|-------------------------------------------------------|
| **Web Application Hosting**  | Streamlit Cloud              | Host the interactive dashboard and allow real-time model interaction. |
| **Model Deployment**         | Pickle/Joblib                | Serialize and load the trained model for inference.    |
| **Feature Interpretability** | SHAP                         | Visualize the impact of each feature on the predicted score. |
| **Real-Time Predictions**    | Streamlit Widgets, ML model  | Allow users to input features and receive predicted specification scores. |
| **Monitoring and Observability** | Prometheus-Grafana Stack  | Monitor application performance, uptime, and user interactions. |



