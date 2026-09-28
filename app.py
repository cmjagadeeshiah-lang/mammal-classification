"""Flask backend that connects the browser frontend (index.html) to the
exported Teachable Machine mammal model, reusing predict.py's functions.

Folder layout expected (same folder as predict.py):
    app.py
    predict.py
    templates/index.html
    converted_keras/keras_model.h5
    converted_keras/labels.txt

Run:
    py -3.10 -m pip install flask
    py -3.10 app.py

Then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import io
import logging
import os
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, render_template, request
from PIL import UnidentifiedImageError

# Keep TensorFlow's startup logging quiet, same as predict.py.
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
logging.getLogger("tensorflow").setLevel(logging.ERROR)

import tensorflow as tf  # noqa: E402  (import after env var is set)

tf.get_logger().setLevel("ERROR")

# Reuse the exact same functions predict.py uses, so behavior stays identical.
from predict import (  # noqa: E402
    DEFAULT_LABELS_PATH,
    DEFAULT_MODEL_PATH,
    load_labels,
    model_class_count,
    model_image_size,
    prepare_image,
)

PROJECT_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

# --- Load the model once at startup, not per-request. ---
print("Loading model, please wait...")
_model = tf.keras.models.load_model(DEFAULT_MODEL_PATH, compile=False)
_class_count = model_class_count(_model)
_labels = load_labels(DEFAULT_LABELS_PATH, _class_count)
_image_size = model_image_size(_model)
print(f"Model loaded. Classes: {_labels}")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "No image file was uploaded."}), 400

    uploaded_file = request.files["image"]
    if uploaded_file.filename == "":
        return jsonify({"error": "No image file was selected."}), 400

    try:
        image_bytes = io.BytesIO(uploaded_file.read())
        image_batch = prepare_image(image_bytes, _image_size)
        probabilities = np.asarray(_model.predict(image_batch, verbose=0))[0]
    except UnidentifiedImageError:
        return jsonify({"error": "That file isn't a readable image."}), 400
    except Exception as error:  # noqa: BLE001 - surface a clean message to the UI
        return jsonify({"error": str(error)}), 500

    ranked_indices = np.argsort(probabilities)[::-1]
    results = [
        {"label": _labels[int(index)], "probability": float(probabilities[int(index)])}
        for index in ranked_indices
    ]

    return jsonify({
        "prediction": results[0]["label"],
        "confidence": results[0]["probability"],
        "results": results,
    })


if __name__ == "__main__":
    app.run(debug=True, port=5000)
