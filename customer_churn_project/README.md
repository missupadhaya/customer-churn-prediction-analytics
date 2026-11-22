# Customer Churn Prediction & Explainable Analytics Dashboard

A portfolio project that demonstrates a machine-learning workflow for estimating customer churn risk, comparing metrics, exploring churn patterns, and serving predictions through a Streamlit app.

> **Dataset notice:** `data/customer_churn.csv` is a reproducibly generated synthetic dataset (not real customer records). Results are demonstration metrics only and should not be presented as validated business performance.

## Features
- Exploratory Data Analysis notebook with plots
- Data preprocessing using a scikit-learn pipeline (imputation, scaling, one-hot encoding)
- Random Forest classifier with class weighting
- Evaluation: Accuracy, Precision, Recall, F1 and ROC-AUC
- Streamlit dashboard for dataset overview and single-customer prediction
- Saved trained pipeline in `models/`


## Tech stack
Python · Pandas · NumPy · Scikit-learn · Streamlit · Matplotlib · Seaborn

## Dataset columns
- `customer_id`: synthetic ID
- `tenure_months`: customer tenure
- `monthly_charges`: monthly subscription amount
- `support_tickets`: number of support tickets
- `service_outages`: service interruption count
- `contract_type`, `internet_service`, `payment_method`: categorical attributes
- `churn`: target (0 = retained, 1 = churned)

## Run locally
1. Install Python 3.10+.
2. Clone/download this repository and open the folder in terminal.
3. Create a virtual environment (optional but recommended):
   - Windows: `python -m venv .venv` then `.venv\Scripts\activate`
   - macOS/Linux: `python3 -m venv .venv` then `source .venv/bin/activate`
4. Install dependencies: `pip install -r requirements.txt`
5. Launch app: `streamlit run app.py`


## Retrain
Run `python train.py` from the repository root to regenerate the model and metrics.


## GitHub upload
Create a new GitHub repository named `customer-churn-prediction-analytics`. Keep it Public if you want recruiters to view it. Upload the project files from this folder (or push via Git). Do not commit credentials or secrets.


## Resume bullet (adapt after she runs and understands it)
- Built an end-to-end customer churn prediction demo using Python and Scikit-learn, including preprocessing, Random Forest modeling, evaluation metrics and a Streamlit interface for interactive risk estimation.


## Example evaluation snapshot

Model metrics for the included fixed synthetic dataset are saved in `models/evaluation_metrics.json`. Re-run `train.py` to reproduce them locally.
