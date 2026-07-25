# House Price Prediction Model Evaluation Report

## Project Overview

This project predicts residential house prices using machine learning regression
models trained on the Ames Housing Dataset.

The workflow includes:
- Data cleaning and preprocessing
- Exploratory Data Analysis
- Feature engineering
- Regression model development
- Model evaluation
- Prediction interface using Streamlit


## Data Preprocessing

The dataset was cleaned by:

- Handling missing values using appropriate imputation methods.
- Encoding categorical variables.
- Selecting relevant numerical and categorical features.
- Preparing the dataset for machine learning.


## Feature Engineering

The engineered-feature definitions are implemented in `feature_engineering.py` for use during retraining.

The following features were created:

| Feature | Description |
|---|---|
| TotalSF | Total usable house area |
| TotalBathrooms | Weighted bathroom count |
| HouseAge | Age of house during sale |
| YearsSinceRemodel | Years after renovation |
| TotalOutdoorSF | Combined outdoor spaces |
| TotalRooms | Total room indicator |

These features improve model understanding of house size,
condition, and usability.


## Models Developed

Three regression algorithms were implemented:

1. Linear Regression
2. Decision Tree Regression
3. Random Forest Regression


## Model Performance

Final Random Forest Model Results:

| Metric | Value |
|---|---:|
| R² Score | 0.897 |
| MAE | $17,747 |
| RMSE | $28,117 |


## Model Selection

Random Forest achieved the best performance because it:

- Handles nonlinear relationships effectively.
- Reduces overfitting through ensemble learning.
- Captures interactions between housing features.


## Prediction Interface

A Streamlit application was developed to allow users to enter property details
and receive predicted house prices.

Application file:

`pages/Predict.py`


## Conclusion

The final Random Forest regression model provides accurate house price
predictions with an R² score of approximately 0.897.

The project successfully demonstrates an end-to-end machine learning pipeline
from preprocessing to deployment.
