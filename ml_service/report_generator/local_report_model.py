# local_report_model.py
"""
Lightweight offline fallback model for generating clinical-style reports.
This ensures report generation works even without Gemini or internet.
"""

def generate_local_report(ct_result, audio_result, metadata, fusion_score):
    age = metadata.get("age", "N/A")
    sex = metadata.get("sex", "N/A")

    report = f"""
## MEDICAL REPORT

**PATIENT:** {age}-year-old {sex}

---

**FINDINGS:**
* CT cancer score: {ct_result.get('cancer_risk')}
* Audio score: {audio_result.get('audio_score')}

**IMPRESSION:**
* Combined fusion risk score: {fusion_score:.3f}.
  This represents an intermediate probability based on CT + audio patterns.

**RECOMMENDATIONS:**
* Clinical correlation recommended.
* Follow-up evaluation or additional diagnostic tests as appropriate.

**CONFIDENCE:**
* Report generated using offline local rule-based model.
"""

    return report.strip()
