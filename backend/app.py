"""
Flask API Server — Explainable Chilli Quality Grading System
Endpoints:
  POST /api/analyse  — upload image, returns full analysis JSON
  GET  /api/health   — health check
  GET  /api/grades   — returns grade definitions
"""

import os
import sys
import io
import base64
import json
import time
import traceback

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from PIL import Image

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from model.chilli_model import load_model, analyse_chilli, CHILLI_TYPES, QUALITY_GRADES

app = Flask(__name__, static_folder="../frontend", static_url_path="")
CORS(app, origins="*")

import numpy as np

class SafeJSONProvider(app.json_provider_class):
    """Handles numpy scalars, numpy bools, and Python bools correctly."""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)

app.json_provider_class = SafeJSONProvider
app.json = SafeJSONProvider(app)

# ─────────────────────────────────────────
# Load Model (once at startup)
# ─────────────────────────────────────────

MODEL_PATH = os.path.join(os.path.dirname(__file__), "chilli_net.pth")
print("🌶  Loading ChilliNet model...")
model = load_model(MODEL_PATH)
print("✅  Model ready.")

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def allowed_file(filename):
    ext = os.path.splitext(filename.lower())[1]
    return ext in ALLOWED_EXTENSIONS


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model": "ChilliNet-MobileNetV2",
        "version": "1.0.0",
        "classes": len(CHILLI_TYPES)
    })


@app.route("/api/grades", methods=["GET"])
def grades():
    return jsonify({
        "types": CHILLI_TYPES,
        "grades": {k: {**v, "score_range": list(v["score_range"])} 
                   for k, v in QUALITY_GRADES.items()}
    })


@app.route("/api/analyse", methods=["POST"])
def analyse():
    start = time.time()

    # ── Validate input ──────────────────────
    if "image" not in request.files:
        return jsonify({"error": "No image file provided. Use key 'image'."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "Empty filename."}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": f"Unsupported format. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"}), 415

    raw = file.read()
    if len(raw) > MAX_FILE_SIZE:
        return jsonify({"error": "File too large. Max 10 MB."}), 413

    # ── Load image ──────────────────────────
    try:
        pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        return jsonify({"error": "Could not decode image. Please upload a valid image file."}), 422

    # Embed original thumbnail as base64 for frontend display
    thumb = pil_img.copy()
    thumb.thumbnail((400, 400))
    buf = io.BytesIO()
    thumb.save(buf, format="JPEG", quality=85)
    thumb_b64 = "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()

    # ── Run analysis ─────────────────────────
    try:
        result = analyse_chilli(pil_img, model, return_gradcam=True)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": f"Analysis failed: {str(e)}"}), 500

    elapsed = round(time.time() - start, 2)
    result["original_image"] = thumb_b64
    result["processing_time_s"] = elapsed
    result["image_size"] = {"width": pil_img.width, "height": pil_img.height}

    return jsonify(result)


# ─────────────────────────────────────────
# Dev entry-point
# ─────────────────────────────────────────

if __name__ == "__main__":
    print("\n🌶  Chilli Quality Grading Server")
    print("   API : http://localhost:5000/api/analyse")
    print("   UI  : http://localhost:5000\n")
    # use_reloader=False prevents the watchdog from restarting the server
    # when it detects changes in torch/flask library files (which caused
    # "Failed to fetch" errors on the frontend).
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)
