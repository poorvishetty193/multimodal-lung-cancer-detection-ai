REPORT_TEMPLATE = """
You are an AI radiology assistant. Based on the following inputs, produce
a short and clinically safe report.

CT Risk Score: {ct_score}

Audio Cough Analysis Score: {audio_score}

Metadata:
- Age: {age}
- Sex: {sex}
- Smoking History: {smoking}
- Symptoms: {symptoms}

Fusion Risk Score: {fusion_score}

Write a final clinical-style summary including:
- Findings
- Impression
- Recommendations
- Confidence & safety notes
"""
