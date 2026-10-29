# streamlit_app/app.py
import streamlit as st
import pandas as pd
import json
from src.config import METRICS_PATH
from src.predict import Predictor

st.set_page_config(page_title="Bank Marketing Intelligence", layout="wide")
st.title("🏦 Bank Deposit Subscription Intelligence")

@st.cache_resource
def get_predictor():
    return Predictor()

predictor = get_predictor()

tab1, tab2, tab3 = st.tabs(["📊 Model Performance", "🎯 Single Customer Prediction", "📁 Batch Prediction (CSV)"])

# TAB 1: Evaluation Metrics
with tab1:
    if METRICS_PATH.exists():
        with open(METRICS_PATH, "r") as f:
            m = json.load(f)
        
        st.subheader("Model Leaderboard (5-Fold Cross-Validation)")
        st.dataframe(pd.DataFrame(m["leaderboard"]), width="stretch")

        col1, col2 = st.columns(2)
        col1.metric("Selected Champion Model", m["best_model"])
        col2.metric("Holdout Test ROC-AUC", f"{m['test_roc_auc']:.4f}")

        st.subheader("Holdout Confusion Matrix")
        st.table(pd.DataFrame(
            m["confusion_matrix"], 
            index=["Actual No (0)", "Actual Yes (1)"], 
            columns=["Pred No (0)", "Pred Yes (1)"]
        ))
    else:
        st.warning("Metrics file missing. Pehle model train karein.")

# TAB 2: Single Customer Form
with tab2:
    st.subheader("Customer Profile Details")
    threshold = st.slider("Campaign Decision Threshold", 0.1, 0.9, 0.35, 0.05)
    
    with st.form("single_prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", 18, 100, 35)
            job = st.selectbox("Job", ["admin.", "technician", "services", "management", "retired", "blue-collar", "unemployed", "entrepreneur", "housemaid", "student", "self-employed", "unknown"])
            marital = st.selectbox("Marital Status", ["married", "single", "divorced", "unknown"])
            education = st.selectbox("Education", ["primary", "secondary", "tertiary", "unknown", "basic.4y", "basic.6y", "basic.9y", "high.school", "university.degree", "professional.course"])
            default = st.selectbox("Credit Default?", ["no", "yes", "unknown"])
        
        with c2:
            balance = st.number_input("Average Yearly Balance (€)", -5000, 100000, 1500)
            housing = st.selectbox("Housing Loan?", ["no", "yes", "unknown"])
            loan = st.selectbox("Personal Loan?", ["no", "yes", "unknown"])
            contact = st.selectbox("Contact Communication", ["cellular", "telephone", "unknown"])
            day = st.slider("Last Contact Day", 1, 31, 15)
        
        with c3:
            month = st.selectbox("Last Contact Month", ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"])
            campaign = st.number_input("Contacts during this Campaign", 1, 50, 1)
            pdays = st.number_input("Days passed after prev campaign (-1 / 999)", -1, 999, -1)
            previous = st.number_input("Contacts before this campaign", 0, 50, 0)
            poutcome = st.selectbox("Outcome of Previous Campaign", ["unknown", "failure", "other", "success", "nonexistent"])

        submitted = st.form_submit_button("Predict Subscription")
        if submitted:
            payload = [{
                "age": age, "job": job, "marital": marital, "education": education,
                "default": default, "balance": balance, "housing": housing, "loan": loan,
                "contact": contact, "day_of_week": day,
                "month": month, "campaign": campaign, "pdays": pdays,
                "previous": previous, "poutcome": poutcome
            }]
            result = predictor.predict(payload, threshold=threshold)[0]
            prob = result["probability"] * 100

            st.write("---")
            if result["subscribe"]:
                st.success(f"🎯 **Prediction: Likely to Subscribe!** (Probability: {prob:.2f}%)")
            else:
                st.info(f"⚪ **Prediction: Unlikely to Subscribe.** (Probability: {prob:.2f}%)")

# TAB 3: Batch CSV Prediction
with tab3:
    st.subheader("Upload Batch CSV for Targeted Marketing")
    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])
    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        clean_df = batch_df.drop(columns=['y', 'duration'], errors='ignore')
        
        records = clean_df.to_dict(orient="records")
        preds = predictor.predict(records)
        
        clean_df["Predicted_Subscribe"] = [p["subscribe"] for p in preds]
        clean_df["Subscription_Probability"] = [p["probability"] for p in preds]
        
        st.write("### Prediction Results Preview")
        st.dataframe(clean_df.head(20), width="stretch")
        
        csv_download = clean_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Full Predictions CSV",
            data=csv_download,
            file_name="bank_campaign_predictions.csv",
            mime="text/csv"
        )