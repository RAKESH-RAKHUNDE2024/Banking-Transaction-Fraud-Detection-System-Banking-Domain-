import os
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, Response, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates


app = FastAPI(title="Banking Fraud Detection Website")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

MODEL_PATH = os.path.join(BASE_DIR, "check.model")

# Logging File (Real Time Industry Style Tracking)
LOGS_DIR = os.path.join(BASE_DIR, "logs")
os.makedirs(LOGS_DIR, exist_ok=True)
PRED_LOG_PATH = os.path.join(LOGS_DIR, "prediction_history.csv")

# Batch Output File
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
BATCH_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "batch_predictions_output.csv")


# Dropdown Values (Oased On Your Dataset)
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
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("❌ check.model not found. Please run: python run_training.py")
    return joblib.load(MODEL_PATH)


model = load_model()


def make_features(payload: dict) -> pd.DataFrame:
    amount_inr = float(payload["amount_inr"])
    hour_of_day = int(payload["hour_of_day"])

    amount_log = float(np.log1p(amount_inr))
    same_bank_transfer = int(payload["sender_bank"] == payload["receiver_bank"])
    is_night = int(0 <= hour_of_day <= 6)

    row = {
        "transaction_type": payload["transaction_type"],
        "merchant_category": payload["merchant_category"],
        "amount_inr": amount_inr,
        "transaction_status": payload["transaction_status"],
        "sender_age_group": payload["sender_age_group"],
        "receiver_age_group": payload["receiver_age_group"],
        "sender_state": payload["sender_state"],
        "sender_bank": payload["sender_bank"],
        "receiver_bank": payload["receiver_bank"],
        "device_type": payload["device_type"],
        "network_type": payload["network_type"],
        "hour_of_day": hour_of_day,
        "day_of_week": payload["day_of_week"],
        "is_weekend": int(payload["is_weekend"]),
        "amount_log": amount_log,
        "same_bank_transfer": same_bank_transfer,
        "is_night": is_night,
    }

    return pd.DataFrame([row])


def risk_bucket(prob: float):
    if prob <= 0.30:
        return "LOW", "low", "Low risk transaction. Looks normal."
    elif prob <= 0.70:
        return "MEDIUM", "medium", "Medium risk. Review transaction details."
    else:
        return "HIGH", "high", "High risk fraud suspected. Immediate action recommended."


def log_prediction(input_payload: dict, label: str, proba: float):
    row = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "transaction_type": input_payload["transaction_type"],
        "merchant_category": input_payload["merchant_category"],
        "amount_inr": float(input_payload["amount_inr"]),
        "transaction_status": input_payload["transaction_status"],
        "sender_state": input_payload["sender_state"],
        "sender_bank": input_payload["sender_bank"],
        "receiver_bank": input_payload["receiver_bank"],
        "device_type": input_payload["device_type"],
        "network_type": input_payload["network_type"],
        "hour_of_day": int(input_payload["hour_of_day"]),
        "day_of_week": input_payload["day_of_week"],
        "is_weekend": int(input_payload["is_weekend"]),
        "label": label,
        "fraud_probability": round(float(proba), 4),
    }

    df_row = pd.DataFrame([row])

    if os.path.exists(PRED_LOG_PATH):
        df_row.to_csv(PRED_LOG_PATH, mode="a", header=False, index=False)
    else:
        df_row.to_csv(PRED_LOG_PATH, index=False)


@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Fraud Prediction",
            "transaction_types": TRANSACTION_TYPES,
            "merchant_categories": MERCHANT_CATEGORIES,
            "transaction_status": TRANSACTION_STATUS,
            "age_groups": AGE_GROUPS,
            "sender_states": SENDER_STATES,
            "banks": BANKS,
            "device_types": DEVICE_TYPES,
            "network_types": NETWORK_TYPES,
            "days": DAYS,
            "result": None,
            "probability": None,
            "risk_level": None,
            "risk_class": None,
            "progress_width": 0,
            "message": None,
        }
    )


@app.post("/predict-ui", response_class=HTMLResponse)
def predict_ui(
    request: Request,
    transaction_type: str = Form(...),
    merchant_category: str = Form(...),
    amount_inr: float = Form(...),
    transaction_status: str = Form(...),
    sender_age_group: str = Form(...),
    receiver_age_group: str = Form(...),
    sender_state: str = Form(...),
    sender_bank: str = Form(...),
    receiver_bank: str = Form(...),
    device_type: str = Form(...),
    network_type: str = Form(...),
    hour_of_day: int = Form(...),
    day_of_week: str = Form(...),
    is_weekend: int = Form(...),
):
    payload = {
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
    }

    X = make_features(payload)
    pred = int(model.predict(X)[0])
    proba = float(model.predict_proba(X)[0][1])

    label = "FRAUD" if pred == 1 else "NORMAL"

    risk_level, risk_class, msg = risk_bucket(proba)
    progress_width = int(round(proba * 100))

    # Log It
    log_prediction(payload, label, proba)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Fraud Prediction",
            "transaction_types": TRANSACTION_TYPES,
            "merchant_categories": MERCHANT_CATEGORIES,
            "transaction_status": TRANSACTION_STATUS,
            "age_groups": AGE_GROUPS,
            "sender_states": SENDER_STATES,
            "banks": BANKS,
            "device_types": DEVICE_TYPES,
            "network_types": NETWORK_TYPES,
            "days": DAYS,
            "result": label,
            "probability": round(proba, 4),
            "risk_level": risk_level,
            "risk_class": risk_class,
            "progress_width": progress_width,
            "message": msg,
        }
    )


@app.get("/batch", response_class=HTMLResponse)
def batch_page(request: Request):
    return templates.TemplateResponse(
        "batch.html",
        {
            "request": request,
            "title": "Batch Upload",
            "batch_message": None,
            "download_link": None,
        }
    )


@app.post("/batch-predict", response_class=HTMLResponse)
async def batch_predict(request: Request, file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

    # Required Columns Check
    required_cols = [
        "transaction_type", "merchant_category", "amount_inr", "transaction_status",
        "sender_age_group", "receiver_age_group", "sender_state",
        "sender_bank", "receiver_bank", "device_type", "network_type",
        "hour_of_day", "day_of_week", "is_weekend"
    ]

    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return templates.TemplateResponse(
            "batch.html",
            {
                "request": request,
                "title": "Batch Upload",
                "batch_message": f"❌ Missing columns in CSV: {missing}",
                "download_link": None,
            }
        )

    # Feature Engineering
    df["amount_log"] = np.log1p(df["amount_inr"])
    df["same_bank_transfer"] = (df["sender_bank"] == df["receiver_bank"]).astype(int)
    df["is_night"] = ((df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 6)).astype(int)

    preds = model.predict(df)
    probas = model.predict_proba(df)[:, 1]

    df["fraud_prediction"] = preds.astype(int)
    df["fraud_probability"] = np.round(probas, 4)
    df["label"] = df["fraud_prediction"].apply(lambda x: "FRAUD" if x == 1 else "NORMAL")

    df.to_csv(BATCH_OUTPUT_PATH, index=False)

    return templates.TemplateResponse(
        "batch.html",
        {
            "request": request,
            "title": "Batch Upload",
            "batch_message": f"Batch prediction completed. Total rows: {len(df)}",
            "download_link": "/download-batch",
        }
    )


@app.get("/download-batch")
def download_batch():
    return FileResponse(
        path=BATCH_OUTPUT_PATH,
        filename="batch_predictions_output.csv",
        media_type="text/csv"
    )


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(request: Request):
    if not os.path.exists(PRED_LOG_PATH):
        total_predictions = 0
        fraud_count = 0
        fraud_rate = 0
        top_transaction_type = "-"
        top_merchant_category = "-"
        top_device_type = "-"
    else:
        df = pd.read_csv(PRED_LOG_PATH)

        total_predictions = int(len(df))
        fraud_count = int((df["label"] == "FRAUD").sum())
        fraud_rate = round((fraud_count / total_predictions) * 100, 2) if total_predictions > 0 else 0

        fraud_df = df[df["label"] == "FRAUD"]

        top_transaction_type = fraud_df["transaction_type"].mode().iloc[0] if len(fraud_df) else "-"
        top_merchant_category = fraud_df["merchant_category"].mode().iloc[0] if len(fraud_df) else "-"
        top_device_type = fraud_df["device_type"].mode().iloc[0] if len(fraud_df) else "-"

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "title": "Dashboard",
            "total_predictions": total_predictions,
            "fraud_count": fraud_count,
            "fraud_rate": fraud_rate,
            "top_transaction_type": top_transaction_type,
            "top_merchant_category": top_merchant_category,
            "top_device_type": top_device_type,
        }
    )


@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request):
    rows = []
    if os.path.exists(PRED_LOG_PATH):
        df = pd.read_csv(PRED_LOG_PATH)
        df = df.tail(25)  # last 25 logs
        rows = df.to_dict(orient="records")[::-1]

    return templates.TemplateResponse(
        "history.html",
        {"request": request, "title": "History", "rows": rows}
    )