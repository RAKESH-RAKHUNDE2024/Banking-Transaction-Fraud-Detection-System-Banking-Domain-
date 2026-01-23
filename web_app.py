import os
import joblib
import numpy as np
import pandas as pd


import plotly.express as px

from fastapi import FastAPI, Request, Form, UploadFile, File
from fastapi.responses import HTMLResponse, Response, FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from datetime import datetime, timedelta


from db import SessionLocal, engine, Base
from models import PredictionLog
from auth import verify_admin, create_access_token, decode_token


# ============================================================
# APP INIT
# ============================================================
app = FastAPI(title="Banking Transactional Fraud Detection")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")

MODEL_PATH = os.path.join(BASE_DIR, "check.model")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)
BATCH_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "batch_predictions_output.csv")
EXPORT_HISTORY_PATH = os.path.join(OUTPUT_DIR, "prediction_history_export.csv")

Base.metadata.create_all(bind=engine)


# ============================================================
# DROPDOWNS
# ============================================================
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


# ============================================================
# MODEL LOAD
# ============================================================
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("check.model not found. Run: python run_training.py")
    return joblib.load(MODEL_PATH)

model = load_model()


# ============================================================
# AUTH HELPERS
# ============================================================
def generate_transaction_id(db):
    """
    Generates next transaction id in pattern:
    TXN0000000001 ... TXN0000250000
    """
    last = db.query(PredictionLog).order_by(PredictionLog.id.desc()).first()

    if last and last.transaction_id:
        try:
            last_num = int(last.transaction_id.replace("TXN", ""))
        except:
            last_num = 0
    else:
        last_num = 0

    new_num = last_num + 1
    return f"TXN{new_num:010d}"

def get_user_from_cookie(request: Request):
    token = request.cookies.get("access_token")
    if not token:
        return None
    try:
        payload = decode_token(token)
        return payload.get("sub")
    except Exception:
        return None

def require_login(request: Request):
    user = get_user_from_cookie(request)
    if not user:
        return RedirectResponse(url="/login", status_code=303)
    return None


# ============================================================
# FEATURE ENGINEERING
# ============================================================
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
        return "LOW", "Low", "Low Risk Transaction. Looks Normal."
    elif prob <= 0.70:
        return "MEDIUM", "Medium", "Medium Risk. Review Transaction Details."
    else:
        return "HIGH", "High", "High Risk Fraud Suspected. Immediate Action Recommended."

def generate_signals(payload: dict) -> list:
    signals = []

    amount = float(payload["amount_inr"])
    hour = int(payload["hour_of_day"])
    txn_status = payload["transaction_status"]
    same_bank = payload["sender_bank"] == payload["receiver_bank"]
    is_weekend = int(payload["is_weekend"]) == 1

    if amount >= 10000:
        signals.append("High Amount Transaction Detected (>= ₹10,000).")

    if txn_status == "FAILED":
        signals.append("Transaction Status Is FAILED (Higher Fraud Probability).")

    if 0 <= hour <= 6:
        signals.append("Night Time Transaction (00:00 to 06:00) Detected.")

    if same_bank:
        signals.append("Sender And Receiver Bank Are Same (Possible Mule Transfers).")

    if is_weekend:
        signals.append("Weekend Transaction Detected (Unusual Timing Behavior).")

    return signals


def recommended_action(prob: float) -> str:
    if prob <= 0.30:
        return "Allow Transaction (Low Risk)."
    elif prob <= 0.70:
        return "Review Recommended (Medium Risk) — Verify Customer & Device Signals."
    else:
        return "BLock / Hold Transaction (High Risk) — Escalate To Fraud Team."

# ============================================================
# DB LOGGING
# ============================================================

def log_prediction_mysql(payload: dict, label: str, proba: float, risk_level: str):
    db = SessionLocal()
    try:
        # Always Auto Generate TXN ID In Backend
        txn_id = generate_transaction_id(db)

        row = PredictionLog(
            transaction_id=txn_id,
            transaction_type=payload["transaction_type"],
            merchant_category=payload["merchant_category"],
            amount_inr=float(payload["amount_inr"]),
            transaction_status=payload["transaction_status"],
            sender_age_group=payload["sender_age_group"],
            receiver_age_group=payload["receiver_age_group"],
            sender_state=payload["sender_state"],
            sender_bank=payload["sender_bank"],
            receiver_bank=payload["receiver_bank"],
            device_type=payload["device_type"],
            network_type=payload["network_type"],
            hour_of_day=int(payload["hour_of_day"]),
            day_of_week=payload["day_of_week"],
            is_weekend=int(payload["is_weekend"]),
            label=label,
            fraud_probability=round(float(proba), 4),
            risk_level=risk_level
        )

        db.add(row)
        db.commit()

        # Return Transaction ID So It Can Show In Prediction Result
        return txn_id

    except Exception as e:
        db.rollback()
        print("MySQL Insert Error:", e)
        return None

    finally:
        db.close()


# ============================================================
# BASIC ROUTES
# ============================================================
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request, "title": "Login"})


@app.post("/login")
def login_action(username: str = Form(...), password: str = Form(...)):
    if not verify_admin(username, password):
        return RedirectResponse(url="/login?error=1", status_code=303)

    token = create_access_token({"sub": username})
    resp = RedirectResponse(url="/", status_code=303)
    resp.set_cookie(key="access_token", value=token, httponly=True, samesite="lax")
    return resp


@app.get("/logout")
def logout():
    resp = RedirectResponse(url="/login", status_code=303)
    resp.delete_cookie("access_token")
    return resp


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Fraud Prediction",
            "active_page": "prediction",
            "user": get_user_from_cookie(request),
            "current_year": datetime.now().year,

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

            "signals": [],
            "action": None,
        }
    )


@app.get("/result", response_class=HTMLResponse)
def result_page(
    request: Request,
    result: str = None,
    probability: float = None,
    risk_level: str = None,
    risk_class: str = None,
    progress_width: int = 0,
    message: str = None,
):
    # Signals/Action Will Be Sent From Predict-UI
    # We Keep Safe Defaults Here
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Fraud Prediction",
            "active_page": "prediction",
            "user": get_user_from_cookie(request),
            "current_year": datetime.now().year,

            "transaction_types": TRANSACTION_TYPES,
            "merchant_categories": MERCHANT_CATEGORIES,
            "transaction_status": TRANSACTION_STATUS,
            "age_groups": AGE_GROUPS,
            "sender_states": SENDER_STATES,
            "banks": BANKS,
            "device_types": DEVICE_TYPES,
            "network_types": NETWORK_TYPES,
            "days": DAYS,

            "result": result,
            "probability": probability,
            "risk_level": risk_level,
            "risk_class": risk_class,
            "progress_width": progress_width,
            "message": message,

            "signals": [],
            "action": None,
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

    # No Transaction_Id From UI
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

    # Label Must Be Created Before Logging
    label = "FRAUD" if pred == 1 else "NORMAL"
    risk_level, risk_class, msg = risk_bucket(proba)
    progress_width = int(round(proba * 100))

    # Quick Signals & Action
    signals = generate_signals(payload)
    action = recommended_action(proba)

    # Log ONLY ONCE And Capture Generated Transaction Id
    saved_txn_id = log_prediction_mysql(payload, label, proba, risk_level)

    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "title": "Fraud Prediction",
            "active_page": "prediction",
            "user": get_user_from_cookie(request),
            "current_year": datetime.now().year,

            "transaction_types": TRANSACTION_TYPES,
            "merchant_categories": MERCHANT_CATEGORIES,
            "transaction_status": TRANSACTION_STATUS,
            "age_groups": AGE_GROUPS,
            "sender_states": SENDER_STATES,
            "banks": BANKS,
            "device_types": DEVICE_TYPES,
            "network_types": NETWORK_TYPES,
            "days": DAYS,

            # Show Generated Transaction Id In Prediction Result Only
            "saved_txn_id": saved_txn_id,

            "result": label,
            "probability": round(proba, 4),
            "risk_level": risk_level,
            "risk_class": risk_class,
            "progress_width": progress_width,
            "message": msg,

            "signals": signals,
            "action": action,
        }
    )


# ============================================================
# BATCH ROUTES (FIXED)
# ============================================================
@app.get("/batch", response_class=HTMLResponse)
def batch_page(request: Request):
    return templates.TemplateResponse(
        "batch.html",
        {"request": request, "title": "Batch Upload", "batch_message": None, "download_link": None}
    )


@app.post("/batch-predict", response_class=HTMLResponse)
async def batch_predict(request: Request, file: UploadFile = File(...)):
    df = pd.read_csv(file.file)

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
                "batch_message": f"Missing columns in CSV: {missing}",
                "download_link": None,
            }
        )

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
            "batch_message": f" Batch prediction completed. Total rows: {len(df)}",
            "download_link": "/download-batch",
        }
    )


@app.get("/download-batch")
def download_batch():
    return FileResponse(BATCH_OUTPUT_PATH, filename="batch_predictions_output.csv", media_type="text/csv")


# ============================================================
# DASHBOARD FILTERS (FIXED)
# ============================================================

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard_page(
    request: Request,
    start_date: str = None,
    end_date: str = None,
    txn_type: str = "ALL",
    fraud_only: str = "ALL"
):
    auth_redirect = require_login(request)
    if auth_redirect:
        return auth_redirect

    db = SessionLocal()

    total_predictions = 0
    fraud_count = 0
    fraud_rate = 0

    chart_type_bar = None
    chart_category_pie = None
    chart_trend_line = None

    try:
        logs = db.query(PredictionLog).order_by(PredictionLog.id.asc()).all()
        if not logs:
            return templates.TemplateResponse(
                "dashboard_filters.html",
                {
                    "request": request,
                    "title": "Dashboard",
                    "active_page": "dashboard",
                    "user": get_user_from_cookie(request),
                    "current_year": datetime.now().year,
                    "total_predictions": 0,
                    "fraud_count": 0,
                    "fraud_rate": 0,
                    "chart_type_bar": None,
                    "chart_category_pie": None,
                    "chart_trend_line": None,
                    "transaction_types": ["ALL"] + TRANSACTION_TYPES,
                }
            )

        df = pd.DataFrame([{
            "label": r.label,
            "transaction_type": r.transaction_type,
            "merchant_category": r.merchant_category,
            "fraud_probability": float(r.fraud_probability),
            "created_at": r.created_at
        } for r in logs])

        df["created_at"] = pd.to_datetime(df["created_at"])

        # FIX: Inclusive End_Date Filter
        if start_date:
            start_dt = pd.to_datetime(start_date)
            df = df[df["created_at"] >= start_dt]

        if end_date:
            end_dt = pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
            df = df[df["created_at"] <= end_dt]

        if txn_type != "ALL":
            df = df[df["transaction_type"] == txn_type]

        # Fraud Filter: ALL / FRAUD / NORMAL
        if fraud_only == "FRAUD":
            df = df[df["label"] == "FRAUD"]
        elif fraud_only == "NORMAL":
            df = df[df["label"] == "NORMAL"]

        total_predictions = int(len(df))
        fraud_count = int((df["label"] == "FRAUD").sum())
        fraud_rate = round((fraud_count / total_predictions) * 100, 2) if total_predictions else 0

        if total_predictions > 0:
    #    BAR: Transaction Type (Multi Color Bars)
            type_counts = df["transaction_type"].value_counts().reset_index()
            type_counts.columns = ["transaction_type", "count"]

            fig_bar = px.bar(
            type_counts,
            x="transaction_type",
            y="count",
            color="transaction_type",          # Different Color Per Bar
            text="count",
            title="Transactions by Type (Multi Color Bar)"
    )
            fig_bar.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=30))
            fig_bar.update_traces(textposition="outside")
            chart_type_bar = fig_bar.to_html(full_html=False, include_plotlyjs=False)

    # DONUT: Merchant Category (Multi Color Slices)
            cat_counts = df["merchant_category"].value_counts().reset_index()
            cat_counts.columns = ["merchant_category", "count"]

            fig_donut = px.pie(
            cat_counts,
            names="merchant_category",
            values="count",
            hole=0.45,                         # Donut Style
            title="Merchant Category Distribution (Donut Chart)"
    )
            fig_donut.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=30))
            chart_category_pie = fig_donut.to_html(full_html=False, include_plotlyjs=False)

    # TREND LINE: Fraud Probability Trend (Colored By Label)
            df_last = df.sort_values("created_at").tail(50).copy()
            df_last["idx"] = range(1, len(df_last) + 1)

            fig_line = px.line(
            df_last,
            x="idx",
            y="fraud_probability",
            color="label",                     # Separate Color Fraud Vs Normal Line
            markers=True,
            title="Fraud Probability Trend (Last 50 Records)"
    )
            fig_line.update_layout(height=420, margin=dict(l=20, r=20, t=60, b=30))
            chart_trend_line = fig_line.to_html(full_html=False, include_plotlyjs=False)

    finally:
        db.close()

    return templates.TemplateResponse(
        "dashboard_filters.html",
        {
            "request": request,
            "title": "Dashboard",
            "active_page": "dashboard",
            "user": get_user_from_cookie(request),
            "current_year": datetime.now().year,
            "total_predictions": total_predictions,
            "fraud_count": fraud_count,
            "fraud_rate": fraud_rate,
            "chart_type_bar": chart_type_bar,
            "chart_category_pie": chart_category_pie,
            "chart_trend_line": chart_trend_line,
            "transaction_types": ["ALL"] + TRANSACTION_TYPES,
        }
    )

# ============================================================
# HISTORY + EXPORT
# ============================================================
@app.get("/history", response_class=HTMLResponse)
def history_page(request: Request):
    auth_redirect = require_login(request)
    if auth_redirect:
        return auth_redirect

    db = SessionLocal()
    rows = []

    try:
        logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).limit(50).all()
        for r in logs:
            rows.append({
    "transaction_id": r.transaction_id,
    "timestamp": r.created_at.strftime("%Y-%m-%d %H:%M:%S") if r.created_at else "-",
    "transaction_type": r.transaction_type,
    "merchant_category": r.merchant_category,
    "amount_inr": r.amount_inr,
    "label": r.label,
    "fraud_probability": r.fraud_probability, 
     })
    finally:
        db.close()

    return templates.TemplateResponse("history.html", {"request": request, "title": "History", "rows": rows})


@app.get("/download-history")
def download_history(request: Request):
    auth_redirect = require_login(request)
    if auth_redirect:
        return auth_redirect

    db = SessionLocal()
    try:
        logs = db.query(PredictionLog).order_by(PredictionLog.id.desc()).all()

        df = pd.DataFrame([{
            "transaction_id": r.transaction_id,
            "created_at": r.created_at,
            "transaction_type": r.transaction_type,
            "merchant_category": r.merchant_category,
            "amount_inr": r.amount_inr,
            "transaction_status": r.transaction_status,
            "device_type": r.device_type,
            "network_type": r.network_type,
            "label": r.label,
            "fraud_probability": r.fraud_probability,
            "risk_level": r.risk_level,
        } for r in logs])

        df.to_csv(EXPORT_HISTORY_PATH, index=False)
        return FileResponse(EXPORT_HISTORY_PATH, filename="prediction_history_export.csv", media_type="text/csv")

    finally:
        db.close()