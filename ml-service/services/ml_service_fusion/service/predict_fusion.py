from flask import Flask, request, jsonify
import time, random
app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json()
    job_id = payload.get("job_id")
    ct = payload.get("ct", {})
    audio = payload.get("audio", {})
    metadata = payload.get("metadata", {})
    # simple fusion: average probs across modalities where available
    def safe_get(p, k, default=0.0):
        return p.get(k, default) if isinstance(p, dict) else default

    cancer_types = ["adenocarcinoma", "squamous", "large_cell", "small_cell"]
    fused = {}
    for c in cancer_types:
        vals = []
        # from ct: aggregate top nodule prob if present
        nodules = ct.get("nodules", []) if isinstance(ct, dict) else []
        if nodules:
            vals.append(max([n.get("nodule_probs", {}).get(c, 0) for n in nodules]))
        # audio
        vals.append(safe_get(audio.get("audio_probs", {}), c, 0))
        # metadata
        vals.append(safe_get(metadata.get("metadata_probs", {}), c, 0))
        fused[c] = sum(vals) / max(1, len(vals))
    # Normalize to sum <= 1
    total = sum(fused.values()) or 1.0
    for k in fused:
        fused[k] = fused[k] / total
    risk_score = max(fused.values())
    time.sleep(0.2)
    return jsonify({"job_id": job_id, "final_probs": fused, "risk_score": risk_score, "fusion_embedding":[random.random() for _ in range(32)]})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8104)
