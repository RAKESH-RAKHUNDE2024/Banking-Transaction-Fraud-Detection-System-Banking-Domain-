# Banking Transactional Fraud Detection (ML + Streamlit App)

## Summary
This Repository Contains An End To End Machine Learning Project For Detecting Fraudulent Banking/UPI Transactions Using Transaction Behavior Patterns, Device And Network Signals, And Customer Attributes. The Solution Includes Data Preprocessing, Feature Engineering, Model Training Pipeline, And A Streamlit Based Web Application For Real Time Fraud Prediction With Probability Output.

---

## Objective
The Primary Objective Of This Project Is To Build A Fraud Detection Model That Can:
- Classify Transactions As **Fraud (1)** Or **Non-Fraud (0)**
- Provide A **fraud probability score** For Each Transaction
- Support Real Time Prediction Through An Interactive Streamlit User Interface

---

## Dataset Overview
The Model Is Trained On A Transaction Dataset With Approximately **2,00,000 Records** And The Following Key Features:

- `transaction_type` (P2P, P2M, Bill Payment, Recharge)
- `merchant_category` (Grocery, Fuel, Shopping, etc.)
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
- **Python** (Project Development)
- **Pandas / NumPy** (Data Processing And Feature Engineering)
- **Scikit-learn** (Model Training, Evaluation, Preprocessing Pipeline)
- **Joblib** (Model Serialization And Loading)
- **Streamlit** (Web Application For Prediction)
- **GitHub** (Version Control And Repository Hosting)
- **Render** *(Planned Deployment)* (Cloud Hosting For Streamlit App)

---

## Project Structure
```bash
Fraud_Transactions_ML/
│
├── streamlit_app.py              # Streamlit UI For Fraud Prediction
├── run_training.py               # Pipeline Runner (Training Entry Point)
├── requirements.txt              # Python Dependencies
├── README.md                     # Project Documentation
│
└── transaction/
    ├── components/               # Data Ingestion, Transformation, Training
    ├── pipeline/                 # Training Pipeline Orchestration
    ├── prediction/               # Prediction Pipeline Used By Streamlit
    ├── config.py                 # Configurations And Paths
    ├── utils.py                  # Utility Helpers (Load/Save Objects)
    ├── logger.py                 # Logging Setup
    └── exception.py              # Custom Exception Handling
```

Approach (Step By Step Workflow)
### 1. Data Understanding & EDA

- Performed Exploratory Analysis To Understand:

- Fraud Distribution And Imbalance

- Fraud Patterns Across Merchant Categories And Transaction Types

- Behavioral Trends Based On Time, Device Type, And Network Type

### 2. Data Cleaning & Preprocessing

#### Key Preprocessing Tasks:

- Standardized Column Names For Consistent Processing

- Handled Categorical Features Using Encoding Strategies

- Scaled Numeric Features Such As Transaction Amount And Time Features

### 3. Feature Engineering

- Additional Features Were Created To Improve Fraud Detection:

- amount_log = Log-transformed Transaction Amount

- same_bank_transfer = Sender And Receiver Bank Match Flag

- is_night = Whether Transaction Occurred During Night Hours

### 4. Model Training & Evaluation

- Built A Supervised Ml Pipeline Using Preprocessing + Classification Model

- Evaluated Using Fraud-relevant Metrics Such As:

- Precision

- Recall

- F1 Score

- ROC-AUC (Where Applicable)

### 5. Streamlit Deployment (Prediction App)

- Developed A Streamlit Interface To:

- Accept Transaction Details As Input

- Predict Fraud/Non Fraud Output

- Show Fraud Probability Score For Better Decision Support

### Key Findings (Insights)

- Transaction Amount And Time Based Features Contribute Strongly To Fraud Prediction.

- Certain Transaction Types And Merchant Categories May Show Higher Fraud Exposure.

- Device And Network Indicators Provide Useful Signals For Abnormal Transaction Behavior.

### Results

### The System Provides:

- Binary Fraud Classification (fraud_flag)

- Fraud Probability Score (predict_proba)

- Real Time Prediction Through Streamlit UI

### Conclusion

This Project Demonstrates A Complete Machine Learning Workflow For Banking/UPI Fraud Detection, Including Data Preparation, Feature Engineering, Model Training, And Interactive Deployment. The Final Output Supports Fraud Risk Identification With Probability Scoring, Enabling Effective Monitoring And Decision-making.
