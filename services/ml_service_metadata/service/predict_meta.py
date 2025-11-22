from flask import Flask, request, jsonify
import time
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    job_id = payload.get("job_id")
    metadata = payload.get("metadata", {})
    # trivial rule-based risk
    age = metadata.get("age", 50) or 50
    pack_years = metadata.get("smoking_history_pack_years", 0) or 0
    base = 0.01 + max(0, (age - 40) * 0.005) + min(0.3, pack_years * 0.002)
    probs = {"adenocarcinoma": base*0.5, "squamous": base*0.3, "large_cell": base*0.1, "small_cell": base*0.1}
    time.sleep(0.3)
    return jsonify({"job_id": job_id, "metadata_probs": probs, "meta_embedding":[base]*16})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8103)
