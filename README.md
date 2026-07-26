# 🏠 SmartHome AI: House Price Prediction

<p align="center">

A machine-learning powered property valuation application built with **Streamlit** that estimates residential sale prices using the Ames Housing Dataset.

</p>

---

## 📌 Overview

**SmartHome AI** is an end-to-end machine learning project that predicts residential property prices based on important house characteristics.

The application combines:

- 🤖 Machine Learning prediction
- 📊 Interactive analytics dashboard
- 🏡 User-friendly property valuation interface
- 📈 Data-driven insights

A trained **Random Forest Regression model** analyzes selected property features and generates an estimated sale price.

> ⚠️ This project is designed for educational and portfolio demonstration purposes. It is a decision-support tool and should not replace professional property appraisal.

---

# ✨ Features

## 🏡 Property Valuation

Users can enter important property details and receive:

- Estimated house value
- Price category
- Model performance information
- Prediction explanation


## 📊 Analytics Dashboard

Explore:

- Sale price distribution
- Feature relationships
- Correlation analysis
- Important factors affecting property prices
- Housing dataset insights


## ⚡ Optimized Application

The application includes:

- Cached model loading
- Reusable utility functions
- Responsive Streamlit layout
- Custom styling
- Error handling and validation

---

# 🧠 Machine Learning Model

### Algorithm Used

**Random Forest Regression**

Why Random Forest?

- Handles complex relationships between housing features
- Works well with structured tabular data
- Reduces overfitting compared to individual decision trees
- Provides feature importance insights


### Model Inputs

The prediction model uses:

| Feature | Description |
|---|---|
| Overall Quality | Overall material and finish quality |
| Overall Condition | Current condition rating |
| Living Area | Above-ground living space |
| Lot Area | Property land size |
| Year Built | Construction year |
| Full Bathrooms | Number of full bathrooms |
| Total Rooms | Rooms above ground |
| Basement Area | Total basement square footage |
| Garage Capacity | Number of cars supported |
| Garage Area | Garage size |


---

# 📂 Project Structure

```
SmartHome-AI/
│
├── 🏠 Home.py
├── style.css
├── utils.py
├── feature_engineering.py
│
├── pages/
│   ├── Analytics.py
│   ├── Predict.py
│   └── About.py
│
├── models/
│   └── house_price_model.pkl
│
├── data/
│   ├── train.csv
│   ├── test.csv
│   └── data_description.txt
│
├── notebooks/
│   └── Model development notebooks
│
├── requirements.txt
└── README.md
```

---

# 🛠️ Tech Stack

### Programming Language

- Python

### Machine Learning

- Scikit-learn
- Random Forest Regression
- Joblib

### Data Processing

- Pandas
- NumPy

### Visualization

- Matplotlib
- Seaborn

### Application

- Streamlit

---

# 🚀 Run Locally

### 1. Clone repository

```bash
git clone https://github.com/navyayenjala-68/house-price-prediction.git
```

### 2. Create virtual environment

```bash
python -m venv venv
```

Activate:

Windows:

```powershell
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Launch application

```bash
streamlit run 🏡Home.py
```

Application will open at:

```
http://localhost:8501
```

---

# 🔄 Project Workflow

```
Data Collection
        ↓
Exploratory Data Analysis
        ↓
Feature Selection
        ↓
Feature Engineering
        ↓
Model Training
        ↓
Model Evaluation
        ↓
Streamlit Deployment
        ↓
User Prediction
```

---

# 📈 Model Evaluation

The trained Random Forest model was evaluated using regression metrics.

Evaluation includes:

- R² Score
- Prediction comparison
- Feature importance analysis


The application displays model performance information along with generated predictions.

---

# 🌐 Deployment

The application is deployed using Streamlit Cloud.

Live Demo:

```
https://house-price-prediction-cxahgy5btxagynsjmkrxka.streamlit.app/
```

GitHub Repository:

```
https://github.com/navyayenjala-68/house-price-prediction.git
```

# ⚠️ Limitations

- Dataset is based on Ames, Iowa housing data.
- Market conditions may differ from current real-world prices.
- Predictions for properties outside the training distribution may be less reliable.
- Estimates should not be used as the only basis for financial decisions.

---

# 👩‍💻 Author

**Navya Yenjala**

Machine Learning | Data Analytics | Streamlit Development


---

