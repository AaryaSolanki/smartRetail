
"""
cv_utils.py
Module A1 - OpenCV Basics

Reusable computer-vision preprocessing functions shared by the rest of the
platform (product classifier, face recognition module, FastAPI vision router).

Covers the Week 6 syllabus points:
    - webcam / video capture
    - grayscale conversion
    - resize
    - blur
    - Canny edge detection
    - Haar Cascade face bounding boxes
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

# --------------------------------------------------------------------------- #
# Haar Cascade is bundled with opencv-python; no separate download needed.
# --------------------------------------------------------------------------- #
_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
_face_cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)


@dataclass
class BoundingBox:
    x: int
    y: int
    w: int
    h: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        return (self.x, self.y, self.w, self.h)


# --------------------------------------------------------------------------- #
# Capture
# --------------------------------------------------------------------------- #
def open_webcam(device_index: int = 0) -> cv2.VideoCapture:
    """Open a webcam / video device. Raises RuntimeError if it can't be opened."""
    cap = cv2.VideoCapture(device_index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video device {device_index}")
    return cap


def frames_from_webcam(device_index: int = 0) -> Iterator[np.ndarray]:
    """Generator that yields BGR frames from a webcam until the caller stops iterating."""
    cap = open_webcam(device_index)
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def read_image(path: str) -> np.ndarray:
    """Read an image from disk as a BGR numpy array."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    img = cv2.imread(path)
    if img is None:
        raise ValueError(f"Could not decode image: {path}")
    return img


def read_image_bytes(data: bytes) -> np.ndarray:
    """Decode raw image bytes (e.g. from an UploadFile in FastAPI) into a BGR array."""
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes")
    return img


# --------------------------------------------------------------------------- #
# Basic preprocessing
# --------------------------------------------------------------------------- #
def to_grayscale(image: np.ndarray) -> np.ndarray:
    if len(image.shape) == 2:
        return image  # already grayscale
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def resize(image: np.ndarray, width: int, height: int, keep_aspect: bool = False) -> np.ndarray:
    if not keep_aspect:
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # pad to exact target size (letterbox)
    canvas = np.zeros((height, width, 3) if len(image.shape) == 3 else (height, width), dtype=image.dtype)
    y_off, x_off = (height - new_h) // 2, (width - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas


def blur(image: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    if kernel_size % 2 == 0:
        kernel_size += 1  # GaussianBlur requires odd kernel size
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def canny_edges(image: np.ndarray, low_threshold: int = 100, high_threshold: int = 200) -> np.ndarray:
    gray = to_grayscale(image)
    return cv2.Canny(gray, low_threshold, high_threshold)


def normalize(image: np.ndarray) -> np.ndarray:
    """Scale pixel values to [0, 1] float32 — typical CNN input prep."""
    return image.astype(np.float32) / 255.0


# --------------------------------------------------------------------------- #
# Face detection (Haar Cascade)
# --------------------------------------------------------------------------- #
def detect_faces(
    image: np.ndarray,
    scale_factor: float = 1.1,
    min_neighbors: int = 5,
    min_size: tuple[int, int] = (30, 30),
) -> list[BoundingBox]:
    """Detect faces and return bounding boxes. Runs on a grayscale copy internally."""
    gray = to_grayscale(image)
    boxes = _face_cascade.detectMultiScale(
        gray,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    return [BoundingBox(int(x), int(y), int(w), int(h)) for (x, y, w, h) in boxes]


def draw_boxes(image: np.ndarray, boxes: list[BoundingBox], color=(0, 255, 0), thickness: int = 2) -> np.ndarray:
    """Return a copy of `image` with bounding boxes drawn (for demo/debug output)."""
    out = image.copy()
    for b in boxes:
        cv2.rectangle(out, (b.x, b.y), (b.x + b.w, b.y + b.h), color, thickness)
    return out


def crop_box(image: np.ndarray, box: BoundingBox, margin: float = 0.0) -> np.ndarray:
    """Crop a bounding box out of an image, optionally with a fractional margin."""
    h_img, w_img = image.shape[:2]
    mx, my = int(box.w * margin), int(box.h * margin)
    x1, y1 = max(box.x - mx, 0), max(box.y - my, 0)
    x2, y2 = min(box.x + box.w + mx, w_img), min(box.y + box.h + my, h_img)
    return image[y1:y2, x1:x2]


# --------------------------------------------------------------------------- #
# Quick manual test: `python cv_utils.py path/to/image.jpg`
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python cv_utils.py <image_path>")
        raise SystemExit(1)

    img = read_image(sys.argv[1])
    gray = to_grayscale(img)
    edges = canny_edges(img)
    faces = detect_faces(img)
    annotated = draw_boxes(img, faces)

    print(f"Image shape: {img.shape}")
    print(f"Faces detected: {len(faces)}")

    cv2.imwrite("out_gray.jpg", gray)
    cv2.imwrite("out_edges.jpg", edges)
    cv2.imwrite("out_faces.jpg", annotated)
    print("Wrote out_gray.jpg, out_edges.jpg, out_faces.jpg")
