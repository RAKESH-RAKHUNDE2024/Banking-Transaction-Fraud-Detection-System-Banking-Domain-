from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from db import Base

class PredictionLog(Base):
    __tablename__ = "prediction_logs"

    id = Column(Integer, primary_key=True, index=True)

    # New Transaction ID
    transaction_id = Column(String(20), unique=True, index=True, nullable=False)

    transaction_type = Column(String(50))
    merchant_category = Column(String(50))
    amount_inr = Column(Float)
    transaction_status = Column(String(20))

    sender_age_group = Column(String(20))
    receiver_age_group = Column(String(20))
    sender_state = Column(String(50))

    sender_bank = Column(String(50))
    receiver_bank = Column(String(50))
    device_type = Column(String(20))
    network_type = Column(String(20))

    hour_of_day = Column(Integer)
    day_of_week = Column(String(20))
    is_weekend = Column(Integer)

    label = Column(String(10))
    fraud_probability = Column(Float)
    risk_level = Column(String(10))

    created_at = Column(DateTime, default=datetime.utcnow)
