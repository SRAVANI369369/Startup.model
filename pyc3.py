import pandas as pd
import numpy as np
import joblib
import os
import streamlit as st

# ML Imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# --- 1. DATA GENERATION & MODEL TRAINING ENGINE ---
MODEL_PATH = "best_startup_model.pkl"

def train_best_model():
    """Generates data, trains 3 models, and saves the best one."""
    # Create Synthetic Dataset
    np.random.seed(42)
    n = 1000
    data = {
        'industry': np.random.choice(['SaaS', 'Fintech', 'AI', 'Health', 'Edtech'], n),
        'funding_amount': np.random.uniform(50000, 5000000, n),
        'funding_rounds': np.random.randint(1, 6, n),
        'founder_exp': np.random.randint(0, 20, n),
        'team_size': np.random.randint(2, 50, n),
        'market_size_bn': np.random.uniform(1, 100, n),
        'rev_growth': np.random.uniform(-0.2, 2.5, n),
        'location': np.random.choice(['SF', 'NY', 'London', 'Berlin', 'Austin', 'Remote'], n)
    }
    df = pd.DataFrame(data)
    
    # Logic: Success = high growth + high exp + some randomness
    df['success'] = ((df['rev_growth'] * 1.5 + df['founder_exp'] * 0.1 + np.random.normal(0, 1, n)) > 1.5).astype(int)

    # Features
    cat_cols = ['industry', 'location']
    num_cols = ['funding_amount', 'funding_rounds', 'founder_exp', 'team_size', 'market_size_bn', 'rev_growth']
    
    X = df.drop('success', axis=1)
    y = df['success']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Preprocessing Pipeline
    preprocessor = ColumnTransformer([
        ('num', Pipeline([('imp', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), num_cols),
        ('cat', Pipeline([('imp', SimpleImputer(strategy='constant', fill_value='N/A')), ('ohe', OneHotEncoder(handle_unknown='ignore'))]), cat_cols)
    ])

    # Model Dictionary
    models = {
        'Logistic Regression': LogisticRegression(),
        'Random Forest': RandomForestClassifier(n_estimators=100),
        'XGBoost': XGBClassifier(eval_metric='logloss')
    }

    best_score = 0
    final_pipeline = None

    # Evaluate and pick best
    for name, model in models.items():
        pipe = Pipeline([('prep', preprocessor), ('clf', model)])
        pipe.fit(X_train, y_train)
        score = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
        if score > best_score:
            best_score = score
            final_pipeline = pipe

    # Save the best one
    joblib.dump(final_pipeline, MODEL_PATH)
    return final_pipeline

# --- 2. STREAMLIT USER INTERFACE ---
def run_app():
    st.set_page_config(page_title="Startup Predictor", layout="wide")
    st.title("🚀 Startup Success Probability Model")
    
    # Load model or train if missing
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Initializing Model..."):
            model = train_best_model()
    else:
        model = joblib.load(MODEL_PATH)

    # Input Layout
    with st.sidebar:
        st.header("Company Profile")
        industry = st.selectbox("Industry", ['SaaS', 'Fintech', 'AI', 'Health', 'Edtech'])
        location = st.selectbox("Location", ['SF', 'NY', 'London', 'Berlin', 'Austin', 'Remote'])
        funding = st.number_input("Total Funding ($)", value=500000)
        rounds = st.slider("Funding Rounds", 1, 10, 2)
        
    col1, col2 = st.columns(2)
    with col1:
        exp = st.number_input("Founder Years of Experience", 0, 40, 5)
        team = st.number_input("Current Team Size", 1, 500, 10)
    with col2:
        market = st.number_input("Market Size ($ Billion)", 0.1, 1000.0, 10.0)
        growth = st.slider("Annual Revenue Growth (%)", -100, 500, 20)

    # Prediction Logic
    if st.button("Calculate Success Probability"):
        input_df = pd.DataFrame([{
            'industry': industry, 'funding_amount': funding, 'funding_rounds': rounds,
            'founder_exp': exp, 'team_size': team, 'market_size_bn': market,
            'rev_growth': growth / 100, 'location': location
        }])
        
        proba = model.predict_proba(input_df)[0][1]
        
        # Result Display
        st.divider()
        st.metric(label="Success Probability", value=f"{proba:.1%}")
        
        if proba > 0.7:
            st.success("High potential! The model sees strong growth indicators.")
        elif proba > 0.4:
            st.warning("Moderate potential. Business fundamentals are stable but risky.")
        else:
            st.error("Low probability. High risk factors detected in current metrics.")

if __name__ == "__main__":
    run_app()