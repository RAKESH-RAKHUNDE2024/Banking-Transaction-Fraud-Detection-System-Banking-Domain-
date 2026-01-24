from sqlalchemy.orm import declarative_base
from sqlalchemy import Column, Integer, String, Float, DateTime, Boolean
from datetime import datetime

Base = declarative_base()

class Transaction(Base):
    __tablename__ = "transactions"

    id = Column(Integer, primary_key=True, index=True)

    transaction_id = Column(String(100), unique=True, nullable=False)
    amount = Column(Float, nullable=False)

    sender_account = Column(String(50), nullable=True)
    receiver_account = Column(String(50), nullable=True)

    is_fraud = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
