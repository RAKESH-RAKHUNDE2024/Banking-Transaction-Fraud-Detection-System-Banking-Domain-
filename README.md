# Banking Transactional Fraud Detection Platform (ML + FastAPI Website)

## Summary
This Project Is An End To End Machine Learning Solution To Detect Fraudulent Banking/UPI Transactions Using Transaction Behavior, Device/Network Signals, And Customer Attributes. It Includes A Complete Training Pipeline, Real Time Fraud Scoring, Batch CSV Fraud Detection, Prediction History Tracking, And A Premium Multi Page FastAPI Website UI For Industry Style Deployment.

---

## Objective
The Main Objective Of This Project Is To Build A Production Ready Fraud Detection System That Can:

- Predict Whether A Transaction Is **FRAUD** Or **NORMAL**
- Provide A **Fraud Probability Score** (`0.0 → 1.0`)
- Display An **Industry Level Web Interface** With Risk Meter And Risk Badges
- Support **Batch Fraud Prediction Via CSV Upload**
- Generate A Downloadable Output File For Business Usage
- Maintain **Prediction History Logs** For Monitoring And Analytics

---

## Key Features (Industry Level)
**Real Time Fraud Prediction (Single Transaction)**
- Input Transaction Details Via Website Form
- Get Fraud Prediction + Probability Score Instantly

**Risk Level Classification**
- LOW / MEDIUM / HIGH Risk Bucket Based On Probability Score
- Risk Meter Progress Bar And Color Badges

**Batch Upload Fraud Prediction**
- Upload CSV Transactions File
- Generate Predictions For All Records
- Download Output CSV With Fraud Scores

**Prediction History Tracking**
- Stores Latest Prediction Logs Automatically
- View Recent Predictions In History Page

**Dashboard Analytics**
- Displays Prediction Summary Counts (Total Predictions, Fraud Count, Fraud %)
- Shows Top Fraud Patterns Based On Logged History

---

## Dataset Overview
The Model Is Trained On A Transaction Dataset With Approximately **250,000 Records**.  
Key Columns Used For Training/Prediction Include:

- `transaction_type` (P2P, P2M, Bill Payment, Recharge)
- `merchant_category` (Grocery, Fuel, Shopping, Etc.)
- `amount_inr`
- `transaction_status` (SUCCESS, FAILED)
- `sender_age_group`, `receiver_age_group`
- `sender_state`
- `sender_bank`, `receiver_bank`
- `device_type` (Android, iOS, Web)
- `network_type` (3G, 4G, 5G, WiFi)
- `hour_of_day`, `day_of_week`, `is_weekend`
- Target: `fraud_flag` (0/1)

---

## Tools & Technologies Used
- **Python**
- **Pandas / NumPy** (Data Processing, Feature Engineering)
- **Scikit-learn** (ML Model Training & Evaluation)
- **Joblib** (Model Persistence)
- **FastAPI** (Backend Web Framework + Routing)
- **Jinja2 Templates** (Multi Page Website Rendering)
- **HTML/CSS** (Premium UI And Layout)
- **Uvicorn** (ASGI Server)
- **GitHub** (Project Hosting & Versioning)
- **Render** (Deployment Ready)

---

## Project Structure
```bash
Fraud_Transactions_ML/
│
├── web_app.py                     # FastAPI Website (Prediction + Batch + Dashboard + History)
├── streamlit_app.py               # Streamlit Prediction App 
├── run_training.py                # Model Training Entry Point
├── requirements.txt               # Dependencies
├── README.md                      # Documentation
├── check.model                    # Trained Model File
│
├── templates/                     # Website Pages
│   ├── base.html
│   ├── index.html
│   ├── batch.html
│   ├── dashboard.html
│   └── history.html
│
├── static/                        # Website Assets
│   └── style.css
│
├── outputs/                       # Batch Prediction Outputs (Auto Created)
│   └── batch_predictions_output.csv
│
├── logs/                          # Prediction History Logs (Auto Created)
│   └── prediction_history.csv
│
└── transaction/                   # ML Pipeline Package
    ├── components/
    ├── pipeline/
    ├── prediction/
    └── ...
```

### Ml Workflow (End To End)
#### 1) Data Understanding & EDA

- Distribution Analysis

- Fraud Patterns Across Categories And Transaction Types

- Device/Network Risk Signals

- Time Based Fraud Trends

#### 2) Data Cleaning & Preprocessing

- Standardized Columns

- Categorical Encoding Via Pipeline

- Numeric Scaling/Transformations

#### 3) Feature Engineering

##### Additional Engineered Features:

- amount_log = Log Transformed Transaction Amount

- same_bank_transfer = Sender And Receiver Bank Match (0/1)

- is_night = Transaction Occurred During Night Hours (0/1)

#### 4) Model Training & Evaluation

- ML Pipeline Training + Evaluation

- Metrics Used For Fraud Performance:

- Precision

- Recall

- F1-Score

- ROC-AUC(Where Applicable)

#### 5) Deployment Ready Website

- FastAPI UI With Prediction + Batch + Dashboard

- Clean UI With Industry Style Theme

### Results & Findings

- Fraud Behavior Shows Strong Influence From Transaction Type, Category, Amount Patterns, And Time Of Transaction.

- Device Type And Network Signals Provide Useful Fraud Indicators For Anomaly Detection.

- Feature Engineering (amount_log, is_night, same_bank_transfer) Improves Model Scoring Stability.


### Conclusion

- This Project Demonstrates A Complete Industry Style ML Solution For Banking Transaction Fraud Detection, Including:

- Data Preprocessing & Feature Engineering

- ML Training Pipeline And Model Creation

- Real Time Fraud Scoring With Probability

- Premium FastAPI Website UI

- Batch Processing Support And Downloadable Results

- Logging & Dashboard Analytics For Monitoring
