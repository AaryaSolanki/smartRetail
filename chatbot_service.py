
"""
chatbot_service.py
Module B3 - Chatbot Basics

Hybrid FAQ chatbot:
    1. Rule-based exact/substring match against intents.json patterns (fast,
       100% precise for known phrasings).
    2. ML fallback: TF-IDF + classifier trained on the same intents.json,
       used whenever the rule-based layer doesn't find a confident match
       (handles paraphrases the rules didn't anticipate).

Deliverable: chatbot_model.pkl (produced by the companion training notebook
04_chatbot_training.ipynb), loaded here at inference time.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass

import joblib

from .nlp_utils import preprocess


@dataclass
class ChatResponse:
    tag: str
    reply: str
    confidence: float
    source: str  # "rule" or "ml"


class ChatbotService:
    def __init__(self, intents_path: str, model_path: str | None = None, ml_confidence_threshold: float = 0.35):
        with open(intents_path, "r") as f:
            self.intents = json.load(f)["intents"]

        self.responses_by_tag = {i["tag"]: i["responses"] for i in self.intents}
        self.ml_confidence_threshold = ml_confidence_threshold

        # Rule-based lookup: cleaned pattern text -> tag
        self._rule_patterns: dict[str, str] = {}
        for intent in self.intents:
            for pattern in intent["patterns"]:
                cleaned = preprocess(pattern)
                if cleaned:
                    self._rule_patterns[cleaned] = intent["tag"]

        self.vectorizer = None
        self.classifier = None
        if model_path:
            self.load_ml_model(model_path)

    def load_ml_model(self, model_path: str) -> None:
        bundle = joblib.load(model_path)
        self.vectorizer = bundle["vectorizer"]
        self.classifier = bundle["classifier"]

    # ----------------------------------------------------------------- #
    # Rule-based layer
    # ----------------------------------------------------------------- #
    def _match_rule(self, cleaned_message: str) -> str | None:
        # 1. exact match on a cleaned pattern
        if cleaned_message in self._rule_patterns:
            return self._rule_patterns[cleaned_message]

        message_tokens = set(cleaned_message.split())
        if not message_tokens:
            return None

        # 2. word-level containment either direction (NOT raw substring — raw
        # substring matching is a trap, e.g. "hi" is a substring of "ship").
        # Restricted to patterns with >= 2 tokens: a single-word pattern like
        # "hey" would otherwise match ANY message that happens to contain
        # that word (e.g. "hey, when will my package arrive" wrongly
        # matching "greeting"). Single-word patterns only match exactly.
        for pattern, tag in self._rule_patterns.items():
            pattern_tokens = set(pattern.split())
            if len(pattern_tokens) < 2:
                continue
            if pattern_tokens <= message_tokens or message_tokens <= pattern_tokens:
                return tag

        # 3. token-overlap match: majority of pattern's words present in message
        # (also restricted to multi-token patterns for the same reason)
        best_tag, best_overlap = None, 0.0
        for pattern, tag in self._rule_patterns.items():
            pattern_tokens = set(pattern.split())
            if len(pattern_tokens) < 2:
                continue
            overlap = len(pattern_tokens & message_tokens) / len(pattern_tokens)
            if overlap > best_overlap:
                best_overlap, best_tag = overlap, tag

        if best_overlap >= 0.6:
            return best_tag
        return None

    # ----------------------------------------------------------------- #
    # ML fallback layer
    # ----------------------------------------------------------------- #
    def _match_ml(self, cleaned_message: str) -> tuple[str, float] | None:
        if self.vectorizer is None or self.classifier is None:
            return None

        X = self.vectorizer.transform([cleaned_message])
        probs = self.classifier.predict_proba(X)[0]
        best_idx = probs.argmax()
        tag = self.classifier.classes_[best_idx]
        confidence = float(probs[best_idx])
        return tag, confidence

    # ----------------------------------------------------------------- #
    # Public API
    # ----------------------------------------------------------------- #
    def get_response(self, message: str) -> ChatResponse:
        cleaned = preprocess(message)

        rule_tag = self._match_rule(cleaned)
        if rule_tag:
            reply = random.choice(self.responses_by_tag.get(rule_tag, self.responses_by_tag["fallback"]))
            return ChatResponse(tag=rule_tag, reply=reply, confidence=1.0, source="rule")

        ml_result = self._match_ml(cleaned)
        if ml_result:
            tag, confidence = ml_result
            if confidence >= self.ml_confidence_threshold:
                reply = random.choice(self.responses_by_tag.get(tag, self.responses_by_tag["fallback"]))
                return ChatResponse(tag=tag, reply=reply, confidence=confidence, source="ml")

        fallback_reply = random.choice(self.responses_by_tag["fallback"])
        return ChatResponse(tag="fallback", reply=fallback_reply, confidence=0.0, source="rule")


# --------------------------------------------------------------------------- #
# CLI demo: python chatbot_service.py "where is my order"
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    INTENTS_PATH = "../../data/intents.json"
    MODEL_PATH = "../models/chatbot_model.pkl"

    import os
    model_path = MODEL_PATH if os.path.exists(MODEL_PATH) else None
    bot = ChatbotService(INTENTS_PATH, model_path)

    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        response = bot.get_response(message)
        print(f"[{response.source} | {response.tag} | conf={response.confidence:.2f}] {response.reply}")
    else:
        print("Type a message (Ctrl+C to quit):")
        while True:
            try:
                message = input("> ")
            except (KeyboardInterrupt, EOFError):
                break
            response = bot.get_response(message)
            print(f"[{response.source} | {response.tag} | conf={response.confidence:.2f}] {response.reply}")
