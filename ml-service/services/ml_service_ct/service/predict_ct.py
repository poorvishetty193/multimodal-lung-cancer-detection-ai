from flask import Flask, request, jsonify
import time, random
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    job_id = payload.get("job_id")
    # simulate CT pipeline
    time.sleep(2)
    # mock lung mask summary and nodules
    nodules = [
        {"x": 50, "y": 60, "z": 30, "diameter_mm": 8, "confidence": 0.87, "nodule_probs": {"adenocarcinoma": 0.6, "squamous":0.2, "large_cell":0.1, "small_cell":0.1}},
        {"x": 120, "y": 80, "z": 45, "diameter_mm": 12, "confidence": 0.92, "nodule_probs": {"adenocarcinoma": 0.2, "squamous":0.7, "large_cell":0.05, "small_cell":0.05}}
    ]
    return jsonify({"job_id": job_id, "nodules": nodules, "lung_mask_summary": {"volume_cc": 2500}, "ct_embedding": [random.random() for _ in range(128)]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8101)
