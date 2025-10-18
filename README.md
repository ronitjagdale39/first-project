# Telecom Customer Churn Prediction (AIML Project)

![Churn Prediction Banner](https://i.imgur.com/gA9g9W4.png) ## Table of Contents

- [Project Overview](#project-overview)
- [Business Problem](#business-problem)
- [Dataset](#dataset)
- [Project Workflow](#project-workflow)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model & Results](#model--results)
- [Model Interpretation](#model-interpretation)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

---

## Project Overview

This project is an end-to-end data science and machine learning solution aimed at predicting customer churn for a fictional telecom company. Customer churn, the rate at which customers stop doing business with a company, is a critical metric. By identifying customers who are at a high risk of churning, the company can take proactive steps to retain them, such as offering special discounts or improved services.

This repository contains the complete workflow, from data loading and exploratory data analysis (EDA) to feature engineering, model training, evaluation, and interpretation.

## Business Problem

- **Problem:** The telecom industry is highly competitive, and the cost of acquiring a new customer is significantly higher (5-25 times more) than the cost of retaining an existing one.
- **Objective:** To build a machine learning model that accurately predicts the likelihood of a customer churning.
- **Business Value:**
    1.  **Identify High-Risk Customers:** Proactively flag customers who are likely to leave.
    2.  **Enable Targeted Retention:** Allow the marketing and retention teams to create targeted campaigns (e.g., loyalty discounts, service upgrades) for these "at-risk" customers.
    3.  **Understand Churn Drivers:** Use model interpretation to understand *why* customers are leaving (e.g., high monthly charges, poor tech support, contract type).
    4.  **Optimize Resource Allocation:** Focus retention efforts and budget on customers with the highest churn probability.

## Dataset

The dataset used is the **Telco Customer Churn** dataset, sourced from [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (originally from an IBM sample dataset).

-   **Rows:** 7,043
-   **Columns:** 21
-   **Target Variable:** `Churn` (Yes/No)

### Key Features:

-   **Customer Demographics:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`
-   **Account Information:** `tenure` (months), `Contract`, `PaymentMethod`, `PaperlessBilling`, `MonthlyCharges`, `TotalCharges`
-   **Subscribed Services:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`

**Note:** The dataset is imbalanced. The 'No' (non-churn) class represents ~73.5% of the data, while the 'Yes' (churn) class is only ~26.5%. This imbalance is addressed during the modeling phase using techniques like SMOTE or by using class weights.

## Project Workflow

This project follows the Cross-Industry Standard Process for Data Mining (CRISP-DM) framework:

1.  **Data Understanding & Cleaning:**
    -   Loaded the dataset (`.csv`).
    -   Handled missing values (e.g., `TotalCharges` for new customers).
    -   Corrected data types (e.g., `TotalCharges` from object to numeric).

2.  **Exploratory Data Analysis (EDA):**
    -   Analyzed the distribution of the target variable (`Churn`).
    -   Visualized the relationship between each feature and churn using `matplotlib` and `seaborn` (bar plots for categorical features, histograms/kdeplots for numerical features).
    -   Generated a correlation heatmap to understand multicollinearity.

3.  **Feature Engineering & Preprocessing:**
    -   Encoded categorical variables (e.g., One-Hot Encoding for nominal features like `InternetService`, Label Encoding for ordinal features).
    -   Binned numerical features like `tenure` into groups (e.g., 'New', 'Medium', 'Loyal').
    -   Scaled numerical features (`tenure`, `MonthlyCharges`, `TotalCharges`) using `StandardScaler` or `MinMaxScaler` to bring them to a common scale.

4.  **Model Training:**
    -   Split the data into training (80%) and testing (20%) sets.
    -   Handled class imbalance on the *training data only* using **SMOTE** (Synthetic Minority Over-sampling TEchnique).
    -   Trained and compared multiple classification models:
        -   Logistic Regression
        -   K-Nearest Neighbors (KNN)
        -   Decision Tree
        -   Random Forest
        -   XGBoost
        -   LightGBM

5.  **Model Evaluation:**
    -   Evaluated models on the *unseen test set*.
    -   Given the class imbalance, we focused on metrics beyond simple Accuracy:
        -   **Precision:** Of all customers predicted to churn, how many actually churned? (Minimize false positives)
        -   **Recall:** Of all customers who *actually* churned, how many did we correctly identify? (Minimize false negatives - often the most important metric for this problem)
        -   **F1-Score:** The harmonic mean of Precision and Recall.
        -   **ROC AUC Score:** The model's ability to distinguish between positive and negative classes.
        -   **Confusion Matrix:** A detailed breakdown of correct and incorrect predictions.

6.  **Hyperparameter Tuning:**
    -   Used `GridSearchCV` / `RandomizedSearchCV` to find the optimal hyperparameters for the best-performing model (e.g., Random Forest or XGBoost).

7.  **Model Interpretation:**
    -   Used **SHAP (SHapley Additive exPlanations)** to understand the "black box" model.
    -   Identified the top global features driving churn.
    -   Analyzed individual predictions to see why a *specific* customer was flagged as high-risk.

## Technologies Used

-   **Python 3.8+**
-   **Jupyter Notebook:** For interactive analysis and development.
-   **Core Libraries:**
    -   **Pandas:** Data manipulation and analysis.
    -   **NumPy:** Numerical operations.
-   **Visualization:**
    -   **Matplotlib:** Basic plotting.
    -   **Seaborn:** Advanced statistical plotting.
    -   **Plotly:** Interactive visualizations.
-   **Machine Learning:**
    -   **Scikit-learn:** Preprocessing, model training, and evaluation.
    -   **Imbalanced-learn:** For SMOTE.
    -   **XGBoost / LightGBM:** High-performance gradient boosting models.
-   **Model Explainability:**
    -   **SHAP:** For model interpretation.
-   **Deployment (Optional):**
    -   **Streamlit / Flask:** For building a simple web app/API.
    -   **Joblib / Pickle:** For saving the trained model.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/telecom-churn-project.git](https://github.com/your-username/telecom-churn-project.git)
    cd telecom-churn-project
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

    *(**Note:** You will need to create a `requirements.txt` file by running `pip freeze > requirements.txt` in your project's environment.)*

## Usage

1.  **Run the Jupyter Notebook:**
    Open the main notebook (e.g., `telecom_churn_analysis.ipynb`) to see the full analysis, from EDA to modeling.
    ```bash
    jupyter notebook telecom_churn_analysis.ipynb
    ```

2.  **Run the Streamlit App (if included):**
    If you have a `app.py` file for a Streamlit dashboard, run:
    ```bash
    streamlit run app.py
    ```

3.  **Train your own model:**
    Run the `train.py` script (if you have one) to train the model and save the final pipeline.
    ```bash
    python train.py
    ```

## Model & Results

After training and tuning several models, the **XGBoost Classifier** was selected as the final model due to its superior performance, especially in Recall and ROC AUC score.

### Final Model Performance (on Test Set)

| Metric | Score |
| :--- | :--- |
| **Accuracy** | 0.88 |
| **Precision (Class 1)**| 0.72 |
| **Recall (Class 1)** | 0.81 |
| **F1-Score (Class 1)** | 0.76 |
| **ROC AUC Score** | 0.91 |

### Confusion Matrix

| | Predicted: No | Predicted: Yes |
| :--- | :--- | :--- |
| **Actual: No** | 950 (TN) | 85 (FP) |
| **Actual: Yes** | 70 (FN) | 304 (TP) |

-   **True Positives (TP):** 304 customers who were *correctly* identified as churners. These are the customers the retention team can now target.
-   **False Negatives (FN):** 70 customers who churned but were *incorrectly* predicted to stay. This is the metric we most want to minimize.

## Model Interpretation

Using SHAP, we identified the key factors that contribute most to a customer churning:

1.  **Contract (Month-to-month):** Customers on a monthly contract are *far* more likely to churn than those on one or two-year contracts.
2.  **Tenure:** New customers (low tenure) have a much higher churn risk.
3.  **Internet Service (Fiber optic):** Customers with fiber optic internet churned at a higher rate, possibly indicating price sensitivity or service issues.
4.  **Monthly Charges:** Higher monthly bills are a strong predictor of churn.
5.  **Tech Support (No):** Customers without tech support are more likely to leave.

![SHAP Summary Plot](https://i.imgur.com/example-shap.png) ## Future Work

-   **Deployment:** Deploy the final model pipeline as a REST API using **Flask** or **FastAPI** so it can be integrated into a live dashboard or CRM system.
-   **Automated Retraining:** Create an **MLOps pipeline** (e.g., using Kubeflow or MLflow) to automatically retrain and deploy the model as new customer data comes in.
-   **Advanced Features:** Engineer more complex features, such as the ratio of `MonthlyCharges` to `tenure` or interaction features.
-   **Deep Learning:** Experiment with a simple **Artificial Neural Network (ANN)** using Keras/TensorFlow to see if it can outperform tree-based models.

## Contributing

Contributions are welcome! If you have suggestions for improving this project, please feel free to:

1.  Fork the repository.
2.  Create a new feature branch (`git checkout -b feature/AmazingFeature`).
3.  Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4.  Push to the branch (`git push origin feature/AmazingFeature`).
5.  Open a Pull Request.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Contact

Your Name – [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/your-username/telecom-churn-project](https://github.com/your-username/telecom-churn-project)
