This project focuses on predicting customer churn in a telecommunications company using machine learning techniques. The dataset contains information about 7,043 customers, their demographics, subscription details, service usage, and whether they churned (left the company) or not.

1. Project Objective

The goal is to build a machine learning model capable of predicting whether a customer is likely to leave the company.
This enables:

Retention strategies to reduce customer churn.

Targeted marketing campaigns for at-risk customers.

Optimized business decision-making.

2. Dataset Overview

The dataset represents a fictional telecom provider in California, providing home phone and internet services.

Key details:

7,043 observations

21 variables

Features include:

Demographics: Gender, SeniorCitizen, Partner, Dependents

Services: Phone, MultipleLines, InternetService, OnlineSecurity, OnlineBackup, DeviceProtection, TechSupport, StreamingTV, StreamingMovies

Account Info: Contract type, PaperlessBilling, PaymentMethod, MonthlyCharges, TotalCharges

Target Variable: Churn (Yes/No)

3. Project Workflow
Step 1 – Exploratory Data Analysis (EDA)

Identified numerical and categorical variables.

Handled data type corrections.

Analyzed distribution of variables and their relationship with Churn.

Checked for missing values and outliers.

Step 2 – Feature Engineering

Treated missing and outlier values.

Created new meaningful variables to improve model performance.

Applied encoding techniques for categorical variables.

Standardized numerical features.

Step 3 – Modeling

Built classification models using:

Logistic Regression

Random Forest

Gradient Boosting (XGBoost, LightGBM)

Support Vector Machines

Compared model performances using Accuracy, Precision, Recall, and F1-score.

Selected top-performing models for hyperparameter tuning.

Step 4 – Model Optimization

Performed GridSearchCV for hyperparameter tuning.

Finalized a highly accurate churn prediction model.

4. Results

Best-performing model achieved:

High accuracy and recall (ensuring we capture most churned customers).

Identified key drivers of churn, including:

Contract type

Tenure (length of stay)

MonthlyCharges

5. Tech Stack

Python: Pandas, NumPy, scikit-learn, XGBoost, LightGBM, Matplotlib, Seaborn

Machine Learning: Classification models, Hyperparameter tuning

EDA & Feature Engineering: Outlier treatment, encoding, scaling

6. Project Structure

├── data/
│   ├── Telco-Customer-Churn.csv      # Dataset
├── notebooks/
│   ├── Telco_Churn_Analysis.ipynb    # EDA & Modeling
├── reports/
│   ├── Telco_Churn_Prediction.pdf    # Project summary (business context)
├── src/
│   ├── preprocessing.py              # Data preprocessing functions
│   ├── modeling.py                   # ML modeling & evaluation
│
├── README.md                         # Project documentation
└── requirements.txt                  # Dependencies

7. How to Run
- Clone the repository
git clone https://github.com/yourusername/telco-churn-prediction.git
cd telco-churn-prediction

- Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate        # (Mac/Linux)
venv\Scripts\activate           # (Windows)

- Install dependencies
pip install -r requirements.txt

- Run the model training script
python src/modeling.py

8. Business Impact

Enables early detection of customers at risk of leaving.

Supports data-driven retention campaigns.

Increases customer lifetime value by reducing churn.

9. Acknowledgements

Dataset source: IBM Sample Telco Churn Dataset
.
