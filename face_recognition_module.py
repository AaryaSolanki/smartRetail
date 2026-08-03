
"""
face_recognition_module.py
Module A3 - Face Recognition Fundamentals

Pipeline: detect face -> generate encoding -> compare against stored customer
encodings -> log visit with timestamp.

Two backends are supported so this works even if the `face_recognition`
(dlib-based) package is hard to install on your machine:

    BACKEND = "face_recognition"  -> more accurate, needs dlib (pip install face_recognition)
    BACKEND = "lbph"              -> OpenCV's built-in LBPH recognizer, pure opencv-contrib-python

Deliverable: face_db.pkl (stored encodings/labels) produced by `enroll_customer`.

Ethics note (include in the report — Module 4/A3):
    This module is for demo/coursework purposes only. Any real deployment
    must (1) get explicit customer consent before enrolling their face,
    (2) disclose the system clearly in-store, (3) allow opt-out/deletion of
    stored biometric data, and (4) be evaluated for accuracy-bias across
    demographic groups before being trusted for anything beyond a loyalty
    nudge. Facial recognition is regulated or restricted in many
    jurisdictions (e.g. Illinois BIPA, EU GDPR biometric-data rules) --
    treat storage of face encodings as sensitive personal data.
"""

from __future__ import annotations

import os
import pickle
import uuid
from dataclasses import dataclass, field
from datetime import datetime

import cv2
import numpy as np

from cv_utils import detect_faces, crop_box, read_image

BACKEND = "lbph"  # "face_recognition" or "lbph" -- see module docstring

if BACKEND == "face_recognition":
    import face_recognition  # type: ignore


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass
class VisitLog:
    customer_id: str
    timestamp: str
    confidence: float


@dataclass
class FaceDB:
    # face_recognition backend: customer_id -> list of 128-d encodings
    encodings: dict[str, list[np.ndarray]] = field(default_factory=dict)
    # lbph backend: needs a trained cv2 recognizer + int-label <-> customer_id map
    label_to_customer: dict[int, str] = field(default_factory=dict)
    visits: list[VisitLog] = field(default_factory=list)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #
def load_db(path: str) -> FaceDB:
    if not os.path.exists(path):
        return FaceDB()
    with open(path, "rb") as f:
        return pickle.load(f)


def save_db(db: FaceDB, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(db, f)


# --------------------------------------------------------------------------- #
# face_recognition (dlib) backend
# --------------------------------------------------------------------------- #
def _encode_face_dlib(image_bgr: np.ndarray) -> np.ndarray | None:
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    locations = face_recognition.face_locations(rgb)
    if not locations:
        return None
    encodings = face_recognition.face_encodings(rgb, locations)
    return encodings[0] if encodings else None


def enroll_customer_dlib(db: FaceDB, customer_id: str, image_paths: list[str]) -> int:
    """Add one or more reference photos for a customer. Returns number of encodings added."""
    added = 0
    for path in image_paths:
        img = read_image(path)
        enc = _encode_face_dlib(img)
        if enc is not None:
            db.encodings.setdefault(customer_id, []).append(enc)
            added += 1
    return added


def recognize_dlib(db: FaceDB, image_bgr: np.ndarray, tolerance: float = 0.5):
    """Return (customer_id, confidence) or (None, 0.0) if no match / no known faces."""
    query_enc = _encode_face_dlib(image_bgr)
    if query_enc is None or not db.encodings:
        return None, 0.0

    best_id, best_dist = None, float("inf")
    for customer_id, encs in db.encodings.items():
        distances = face_recognition.face_distance(encs, query_enc)
        min_dist = float(np.min(distances))
        if min_dist < best_dist:
            best_dist, best_id = min_dist, customer_id

    if best_dist <= tolerance:
        confidence = max(0.0, 1.0 - best_dist)
        return best_id, confidence
    return None, 0.0


# --------------------------------------------------------------------------- #
# LBPH (OpenCV-only) backend — no dlib dependency, easier to install
# --------------------------------------------------------------------------- #
def _face_crop_gray(image_bgr: np.ndarray, size=(200, 200)) -> np.ndarray | None:
    boxes = detect_faces(image_bgr)
    if not boxes:
        return None
    crop = crop_box(image_bgr, boxes[0], margin=0.15)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size)


def _new_lbph_recognizer():
    # requires opencv-contrib-python (not plain opencv-python)
    return cv2.face.LBPHFaceRecognizer_create()


def train_lbph(db: FaceDB, customer_images: dict[str, list[str]], model_out: str) -> None:
    """
    customer_images: {customer_id: [image_path, ...]}
    Trains an LBPH recognizer over ALL enrolled customers and writes:
        - model_out (the trained recognizer, .yml)
        - db.label_to_customer (int label -> customer_id), stored in face_db.pkl
    """
    faces, labels = [], []
    label_to_customer: dict[int, str] = {}

    for label, (customer_id, paths) in enumerate(customer_images.items()):
        label_to_customer[label] = customer_id
        for path in paths:
            img = read_image(path)
            face = _face_crop_gray(img)
            if face is not None:
                faces.append(face)
                labels.append(label)

    if not faces:
        raise ValueError("No faces found in the provided enrollment images.")

    recognizer = _new_lbph_recognizer()
    recognizer.train(faces, np.array(labels))
    recognizer.write(model_out)

    db.label_to_customer = label_to_customer


def recognize_lbph(image_bgr: np.ndarray, recognizer_path: str, db: FaceDB, max_distance: float = 70.0):
    """
    LBPH returns a *distance* (lower = more confident), not a similarity score,
    so we invert it into a rough 0-1 confidence for API consistency with the
    dlib backend.
    """
    if not os.path.exists(recognizer_path) or not db.label_to_customer:
        return None, 0.0

    face = _face_crop_gray(image_bgr)
    if face is None:
        return None, 0.0

    recognizer = _new_lbph_recognizer()
    recognizer.read(recognizer_path)
    label, distance = recognizer.predict(face)

    if distance <= max_distance and label in db.label_to_customer:
        confidence = max(0.0, 1.0 - distance / max_distance)
        return db.label_to_customer[label], confidence
    return None, 0.0


# --------------------------------------------------------------------------- #
# Shared: visit logging (backend-agnostic)
# --------------------------------------------------------------------------- #
def log_visit(db: FaceDB, customer_id: str, confidence: float) -> VisitLog:
    entry = VisitLog(
        customer_id=customer_id,
        timestamp=datetime.utcnow().isoformat(),
        confidence=round(confidence, 4),
    )
    db.visits.append(entry)
    return entry


def new_customer_id() -> str:
    return f"cust_{uuid.uuid4().hex[:8]}"


# --------------------------------------------------------------------------- #
# CLI demo:
#   python face_recognition_module.py enroll <customer_id> <img1> [img2 ...]
#   python face_recognition_module.py recognize <query_img>
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    DB_PATH = "../models/face_db.pkl"
    LBPH_MODEL_PATH = "../models/lbph_model.yml"

    if len(sys.argv) < 2:
        print(__doc__)
        raise SystemExit(1)

    command = sys.argv[1]
    db = load_db(DB_PATH)

    if command == "enroll" and BACKEND == "face_recognition":
        customer_id, *paths = sys.argv[2:]
        n = enroll_customer_dlib(db, customer_id, paths)
        save_db(db, DB_PATH)
        print(f"Enrolled {n} encoding(s) for {customer_id}")

    elif command == "recognize" and BACKEND == "face_recognition":
        img = read_image(sys.argv[2])
        customer_id, confidence = recognize_dlib(db, img)
        if customer_id:
            log_visit(db, customer_id, confidence)
            save_db(db, DB_PATH)
            print(f"Recognized {customer_id} (confidence={confidence:.2f}) — visit logged")
        else:
            print("No matching customer found (new/unknown visitor).")

    else:
        print(f"Backend '{BACKEND}' selected — use the training notebook "
              f"(02_face_recognition_setup.ipynb) to build/train the LBPH model, "
              f"or set BACKEND='face_recognition' for the CLI enroll/recognize flow above.")
