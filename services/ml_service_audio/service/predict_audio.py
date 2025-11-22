from flask import Flask, request, jsonify
import time, random
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    job_id = payload.get("job_id")
    time.sleep(1)
    # mock audio probs for 4 classes and anomaly score
    probs = {"adenocarcinoma": 0.3, "squamous": 0.1, "large_cell":0.05, "small_cell":0.05, "anomaly_score":0.4}
    return jsonify({"job_id": job_id, "audio_probs": probs, "audio_embedding":[random.random() for _ in range(64)]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8102)
