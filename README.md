# Banking Transactional Fraud Detection (ML + Web App)

A Complete End To End Machine Learning Project To Detect Fraudulent Banking/UPI Transactions Using Transaction Behavior, Device/Network Signals, And Customer Attributes.  
This Project Includes A Full ML Training Pipeline, Batch Prediction Support, A Modern Streamlit App, And An Industry Style FastAPI Website With Authentication, MySQL Logging, Dashboard Analytics, And History Export.

---

## Project Summary

Fraudulent Banking And UPI Transactions Are Increasing Due To Fast Digital Adoption And Complex Customer Behavior Patterns.  
This Project Helps Identify Suspicious Transactions Using Historical Transaction Patterns And Behavioral Signals Such As:

- Transaction Type And Merchant Category
- Transaction Amount And Status
- Sender/Receiver Demographics (Age Group, State, Bank)
- Device Type And Network Type
- Time-based Signals (Hour, Weekday, Weekend)

The Model Predicts Whether A Transaction Is Likely **FRAUD** Or **NORMAL**, And Provides A **Fraud Probability Score** For Real Time Decision Making.

---

## Objective

Build A Production Style Fraud Detection System That Supports:

- Data Preprocessing + Feature Engineering  
- Model Training Pipeline (Artifacts + Logs + Saved Model)  
- Single Transaction Fraud Prediction (Real Time UI)  
- Batch Prediction Using CSV Upload  
- Fraud Analytics Dashboard (Filters + Plotly Charts)  
- Prediction History Stored In MySQL Database  
- CSV Export For Prediction Logs  
- Secure Login/Logout System Using JWT Cookies  
- Deploy Ready Structure For Render Cloud

---

## Dataset Information

**File:** `transactions.csv`  
**Rows:** 250,000  

### Key Columns

- `transaction_id`
- `timestamp`
- `transaction_type` (P2P, P2M, Bill Payment, Recharge)
- `merchant_category` (Food, Grocery, Fuel, Shopping, etc.)
- `amount_inr`
- `transaction_status` (SUCCESS / FAILED)
- `sender_age_group`, `receiver_age_group`
- `sender_state`
- `sender_bank`, `receiver_bank`
- `device_type` (Android, iOS, Web)
- `network_type` (3G, 4G, 5G, WiFi)
- `fraud_flag` (Target)
- Time Engineered Columns: `hour_of_day`, `day_of_week`, `is_weekend`

---

## Feature Engineering Used

The Following Engineered Features Were Generated For Improved Fraud Detection Accuracy:

- `amount_log = log1p(amount_inr)`
- `same_bank_transfer = (sender_bank == receiver_bank)`
- `is_night = 1 if hour_of_day between 00 to 06 else 0`

---

## Tools & Technologies Used

### Programming & Libraries
- Python
- Pandas, NumPy
- Scikit Learn
- Joblib
- Plotly (Dashboard Charts)

### Deployment & Web
- Streamlit (ML Prediction App)
- FastAPI (Website + API Layer)
- Jinja2 Templates (HTML UI)
- CSS (Modern Multicolor UI Styling)
- Uvicorn (ASGI Server)

### Database
- MySQL (Prediction Logs Storage)
- SQLAlchemy ORM

---

## Project Folder Structure

```bash
fraud-transactions-ml/
│
├── .env
├── .gitignore
├── README.md
├── requirements.txt
├── check.model
├── transactions.csv
├── streamlit_app.py
├── web_app.py
│
├── notebook/
│   └── EDA_transaction.ipynb
│
├── templates/
│   ├── base.html
│   ├── index.html
│   ├── batch.html
│   ├── dashboard_filters.html
│   ├── history.html
│   └── login.html
│
├── static/
│   └── style.css
│
├── outputs/
│   ├── batch_predictions_output.csv
│   └── prediction_history_export.csv
│
└── transaction/
    ├── __init__.py
    ├── config.py
    ├── exception.py
    ├── logger.py
    ├── utils.py
    │
    ├── entity/
    │   └── artifact_entity.py
    │
    ├── components/
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   ├── model_trainer.py
    │   └── model_evaluation.py
    │
    ├── pipeline/
    │   └── training_pipeline.py
    │
    └── prediction/
        └── prediction_pipeline.py
```

### Website Features

- Prediction UI (Real Time Fraud Scoring)
- Batch Upload & CSV Download
- Analytics Dashboard (Filters + Charts)
- History Tab With MySQL Stored Logs
- Download History As CSV
- Login / Logout Support

### MySQL Integration

All Predictions Are Stored In MySQL In A Table:

#### - Prediction_Logs

#### This Table Stores:

- Transaction_Id (Auto Generated)
- Transaction Features
- Label (FRAUD/ NORMAL)
- Fraud_probability
- Risk_Level
- Timestamp

### Dashboard Filters Supported

#### Dashboard supports filtering by:

- Date Range (start_date / end_date)
- Transaction Type (ALL / P2P / P2M / Bill Payment / Recharge)
- Fraud Filter (ALL / FRAUD / NORMAL)

#### Charts Included:

- Transactions By Type
- Merchant Category Distribution
- Fraud Probability Trend (Last N Predictions)

### Results & Findings
#### This Project Successfully Demonstrated:

- Fraud Probability Scoring For Real Time Decisioning
- Behavioral Signals Detection (Night txn, Failed txn, High Amount)
- Logging System For Audit & Monitoring
- UI Based Fraud Analytics With Filters And Export
- Production Style FastAPI + ML + DB Integration

### Conclusion

- This End To End System Provides A Strong Foundation For Building Fraud Detection Products In A Real Banking/Fintech Environment.
- With Real Time Prediction Support, Batch Scoring, And Dashboard Monitoring, It Aligns Well With Industry Workflows
- Such As Fraud Screening, Transaction Monitoring, And Investigation Support.


### Author

**Rakesh N. Rakhunde**
- **Data Analyst | Data Engineer | ML Enthusiast**
- **End To End Fraud Detection Project (UPI + Banking)**
