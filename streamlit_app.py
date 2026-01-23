import os
import numpy as np
import streamlit as st

st.set_page_config(page_title="Fraud Detection", layout="centered")

st.title("Banking Transactional Fraud Detection (Fixed Streamlit App)")
st.write("Fill transaction details below and click **Predict Fraud**.")

# =========================================================
# Hardcoded dropdown values (100% match your dataset)
# =========================================================
TRANSACTION_TYPES = ["P2P", "P2M", "Bill Payment", "Recharge"]

MERCHANT_CATEGORIES = [
    "Entertainment", "Grocery", "Fuel", "Shopping", "Food",
    "Other", "Utilities", "Transport", "Healthcare", "Education"
]

TRANSACTION_STATUS = ["SUCCESS", "FAILED"]

AGE_GROUPS = ["18-25", "26-35", "36-45", "46-55", "56+"]

SENDER_STATES = [
    "Delhi", "Uttar Pradesh", "Karnataka", "Telangana", "Maharashtra",
    "Gujarat", "Rajasthan", "Tamil Nadu", "West Bengal", "Andhra Pradesh"
]

BANKS = ["Axis", "ICICI", "Yes Bank", "IndusInd", "HDFC", "Kotak", "SBI", "PNB"]

DEVICE_TYPES = ["Android", "iOS", "Web"]

NETWORK_TYPES = ["3G", "4G", "5G", "WiFi"]

DAYS_OF_WEEK = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


# =========================================================
# UI Inputs
# =========================================================
transaction_type = st.selectbox("Transaction Type", TRANSACTION_TYPES)
merchant_category = st.selectbox("Merchant Category", MERCHANT_CATEGORIES)

amount_inr = st.number_input("Amount (INR)", min_value=1, value=500, step=1)

transaction_status = st.selectbox("Transaction Status", TRANSACTION_STATUS)

sender_age_group = st.selectbox("Sender Age Group", AGE_GROUPS, index=1)
receiver_age_group = st.selectbox("Receiver Age Group", AGE_GROUPS, index=0)

sender_state = st.selectbox("Sender State", SENDER_STATES)

sender_bank = st.selectbox("Sender Bank", BANKS)
receiver_bank = st.selectbox("Receiver Bank", BANKS)

device_type = st.selectbox("Device Type", DEVICE_TYPES)
network_type = st.selectbox("Network Type", NETWORK_TYPES)

hour_of_day = st.slider("Hour of Day", 0, 23, 12)
day_of_week = st.selectbox("Day of Week", DAYS_OF_WEEK)
is_weekend = st.selectbox("Is Weekend?", [0, 1])

# =========================================================
# Feature Engineering (same like EDA)
# =========================================================
amount_log = float(np.log1p(amount_inr))
same_bank_transfer = int(sender_bank == receiver_bank)
is_night = int(0 <= hour_of_day <= 6)

# =========================================================
# Build input for model (final features)
# =========================================================
input_data = {
    "transaction_type": transaction_type,
    "merchant_category": merchant_category,
    "amount_inr": amount_inr,
    "transaction_status": transaction_status,
    "sender_age_group": sender_age_group,
    "receiver_age_group": receiver_age_group,
    "sender_state": sender_state,
    "sender_bank": sender_bank,
    "receiver_bank": receiver_bank,
    "device_type": device_type,
    "network_type": network_type,
    "hour_of_day": hour_of_day,
    "day_of_week": day_of_week,
    "is_weekend": is_weekend,

    # engineered features
    "amount_log": amount_log,
    "same_bank_transfer": same_bank_transfer,
    "is_night": is_night
}

with st.expander("Show Input Sent to Model (Debug)"):
    st.write(input_data)

# =========================================================
# Prediction Button
# =========================================================
if st.button("Predict Fraud"):
    st.subheader("Prediction Result")

    # 1) Check model file exists
    model_path = "check.model"
    if not os.path.exists(model_path):
        st.error("Model file not found: check.model")
        st.info("First run model training using: python run_training.py")
        st.stop()

    # 2) Import PredictionPipeline safely
    try:
        from transaction.prediction.prediction_pipeline import PredictionPipeline
    except Exception as e:
        st.error("PredictionPipeline import failed.")
        st.info("Make sure you have these files:")
        st.code(
            "transaction/__init__.py\n"
            "transaction/prediction/__init__.py\n"
            "transaction/prediction/prediction_pipeline.py"
        )
        st.exception(e)
        st.stop()

    # 3) Run Prediction
    try:
        pipeline = PredictionPipeline()
        pred, proba = pipeline.predict(input_data)

        if pred == 1:
            st.error(f"Fraud Detected | Probability: {proba:.2f}")
        else:
            st.success(f"Normal Transaction | Fraud Probability: {proba:.2f}")

    except Exception as e:
        st.error("Prediction failed due to an internal error.")
        st.exception(e)
