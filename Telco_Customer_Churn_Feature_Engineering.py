##############################
# Telco Customer Churn Feature Engineering
##############################

# Problem: The objective is to develop a machine learning model 
# that can predict which customers are likely to churn (leave the company).
# Before building the model, we perform Exploratory Data Analysis (EDA) 
# and Feature Engineering steps.

# Dataset: Telco Customer Churn data includes information 
# about 7043 customers of a fictional telecom company in California, 
# providing home phone and Internet services.
# It contains demographic info, account details, services subscribed, 
# and whether the customer churned.

# 21 Variables, 7043 Observations

# Feature Information:
# CustomerId : Unique customer ID
# Gender : Gender
# SeniorCitizen : Whether the customer is a senior citizen (1, 0)
# Partner : Whether the customer has a partner (Yes, No)
# Dependents : Whether the customer has dependents (Yes, No)
# tenure : Number of months the customer has stayed with the company
# PhoneService : Whether the customer has phone service (Yes, No)
# MultipleLines : Whether the customer has multiple lines (Yes, No, No phone service)
# InternetService : Type of internet service (DSL, Fiber optic, No)
# OnlineSecurity : Whether the customer has online security (Yes, No, No internet service)
# OnlineBackup : Whether the customer has online backup (Yes, No, No internet service)
# DeviceProtection : Whether the customer has device protection (Yes, No, No internet service)
# TechSupport : Whether the customer has tech support (Yes, No, No internet service)
# StreamingTV : Whether the customer streams TV (Yes, No, No internet service)
# StreamingMovies : Whether the customer streams movies (Yes, No, No internet service)
# Contract : Contract type (Month-to-month, One year, Two year)
# PaperlessBilling : Whether the customer has paperless billing (Yes, No)
# PaymentMethod : Payment method (Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic))
# MonthlyCharges : Monthly amount charged to the customer
# TotalCharges : Total amount charged to the customer
# Churn : Whether the customer churned (Yes, No)

# Each row represents one customer.
# Variables contain information about:
# - Subscribed services: phone, internet, online security, device protection, etc.
# - Account info: contract, billing, tenure, charges
# - Demographics: gender, age, partner, dependents


# TASK 1: EXPLORATORY DATA ANALYSIS (EDA)
# Step 1: Examine the general structure.
# Step 2: Identify categorical and numerical variables.
# Step 3: Analyze numerical and categorical variables.
# Step 4: Target variable analysis (Churn).
# Step 5: Outlier analysis.
# Step 6: Missing value analysis.
# Step 7: Correlation analysis.

# TASK 2: FEATURE ENGINEERING
# Step 1: Handle missing and outlier values.
# Step 2: Create new meaningful features.
# Step 3: Perform encoding (Label Encoding, One-Hot Encoding).
# Step 4: Standardize numerical variables.
# Step 5: Build machine learning models.


# Required Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 170)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# Load Dataset
df = pd.read_csv("datasets/Telco-Customer-Churn.csv")

# Convert TotalCharges into numeric (some are strings due to missing values)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors='coerce')

# Encode target variable: Yes → 1, No → 0
df["Churn"] = df["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

##################################
# TASK 1: EXPLORATORY DATA ANALYSIS
##################################

# (All function comments already explain usage in English below...)

# --- GENERAL STRUCTURE CHECK
def check_df(...):
    ...

# --- IDENTIFY CATEGORICAL & NUMERICAL VARIABLES
def grab_col_names(...):
    ...

# --- CATEGORICAL VARIABLE SUMMARY
def cat_summary(...):
    ...

# --- NUMERICAL VARIABLE SUMMARY
def num_summary(...):
    ...

# --- NUMERICAL VARIABLES vs TARGET
def target_summary_with_num(...):
    ...

# --- CATEGORICAL VARIABLES vs TARGET
def target_summary_with_cat(...):
    ...

# --- CORRELATION MATRIX
sns.heatmap(...)

##################################
# TASK 2: FEATURE ENGINEERING
##################################

# --- MISSING VALUE HANDLING
def missing_values_table(...):
    ...

df["TotalCharges"].fillna(df["TotalCharges"].median(), inplace=True)

# --- OUTLIER HANDLING
def outlier_thresholds(...):
    ...
def check_outlier(...):
    ...
def replace_with_thresholds(...):
    ...

# --- NEW FEATURE CREATION
# Example: tenure grouped into years
df.loc[(df["tenure"]>=0) & (df["tenure"]<=12),"NEW_TENURE_YEAR"] = "0-1 Year"
...

# NEW_Engaged: Customers with 1 or 2-year contracts
# NEW_noProt: Customers without protection/backup/tech support
# NEW_Young_Not_Engaged: Younger, non-engaged customers
# NEW_TotalServices: Count of subscribed services
# NEW_FLAG_ANY_STREAMING: Customers using any streaming
# NEW_FLAG_AutoPayment: Automatic payment flag
# NEW_AVG_Charges: Average monthly charges
# NEW_Increase: Ratio of average charges to current monthly charge
# NEW_AVG_Service_Fee: Monthly charge per service

##################################
# ENCODING
##################################

# Label Encoding for binary categories
# One-Hot Encoding for multi-class categories

##################################
# MODELING
##################################

# Train/Test split
# CatBoost Classifier as baseline model
# Print metrics: Accuracy, Recall, Precision, F1, AUC

# Example output:
# Accuracy: 0.80
# Recall: 0.66
# Precision: 0.51
# F1: 0.58
# AUC: 0.75
