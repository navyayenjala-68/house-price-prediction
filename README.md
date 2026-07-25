# SmartHome AI: House Price Prediction

**Internship project submission** — a Streamlit application that demonstrates an end-to-end machine-learning workflow for estimating residential sale prices in Ames, Iowa.

SmartHome AI turns ten practical home attributes into a predicted sale price using a trained Random Forest regression model. The project also includes an analytics dashboard so stakeholders can explore the data and understand the factors associated with price.

## Business objective

Create a simple, explainable prototype that helps a user form an initial property-value estimate. The application is designed as a decision-support tool for learning and portfolio demonstration; it is not a replacement for a licensed appraisal.

## What is included

- A focused property-price prediction form
- An analytics dashboard with price distribution, correlations, feature comparisons, and model importance
- A trained model and Ames Housing source data for local reproducibility
- Shared styling, reusable loading helpers, and clear model-use guidance

## Model inputs

The model uses overall quality and condition, above-ground living area, lot area, year built, full bathrooms, rooms above ground, basement area, garage capacity, and garage area.

## Project structure

| Path | Purpose |
| --- | --- |
| `app.py` | Application home page |
| `pages/` | Estimate, analytics, and project-information pages |
| `models/house_price_model.pkl` | Trained Random Forest model |
| `data/train.csv` | Ames Housing training data |
| `notebooks/` | Exploratory analysis and model-development notebook |
| `feature_engineering.py` | Reusable engineered-feature transformation for model retraining |
| `utils.py` | Shared loading, styling, and layout helpers |

## Run locally

```powershell
# Python 3.10 or newer is required.
py -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open the local address shown in the terminal (normally `http://localhost:8501`).

## Project workflow

1. Explore the Ames Housing training data.
2. Select practical features with a strong relationship to sale price.
3. Engineer interpretable total-area, bathroom, age, renovation, and outdoor-space features for model experiments.
4. Train a Random Forest regression model and save the fitted estimator.
5. Present the selected model through a user-friendly Streamlit interface.
6. Communicate limitations and encourage professional validation.

## Validation notes

- The prediction form verifies that the saved model expects the same number of inputs before generating a result.
- The analytics page handles models that do not expose feature-importance values.
- Data and model loading are cached for a responsive local experience.

## Note

This project is intended for portfolio demonstration and preliminary exploration. The training data covers homes built from 1872 through 2010. The form accepts years through 2025, but estimates for homes built after 2010 are extrapolations and should be validated against current local market evidence. Model estimates should not be used as the sole basis for financial or property decisions.
