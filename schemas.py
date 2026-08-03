"""
Pydantic models for request/response validation across all endpoints.
"""
from pydantic import BaseModel, Field
from typing import Optional, List


# ---------- Vision ----------

class FaceRecognitionResponse(BaseModel):
    recognized: bool
    customer_id: Optional[str] = None
    confidence: Optional[float] = None
    is_new_visit_logged: bool = False
    message: str


class ProductClassificationResponse(BaseModel):
    category: str
    confidence: float
    all_scores: Optional[dict] = None


# ---------- NLP ----------

class SentimentRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, json_schema_extra={"example": "This product is amazing!"})


class SentimentResponse(BaseModel):
    label: str          # Positive / Negative / Neutral
    confidence: float
    cleaned_text: str


# ---------- Chatbot ----------

class ChatbotRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=1000, json_schema_extra={"example": "What is your return policy?"})
    session_id: Optional[str] = "default"


class ChatbotResponse(BaseModel):
    reply: str
    intent: Optional[str] = None
    source: str          # "rule" or "ml"
    confidence: Optional[float] = None


# ---------- Dashboard ----------

class DashboardStats(BaseModel):
    total_visits: int
    unique_customers: int
    sentiment_breakdown: dict
    chatbot_messages_handled: int
    product_classifications: int
    models_loaded: dict
