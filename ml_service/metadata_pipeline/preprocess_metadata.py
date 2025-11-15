# preprocess_metadata.py
"""
Preprocess patient metadata into a clean numerical vector.

Expected metadata fields:
- age (int)
- sex (string: "male", "female", "other")
- smoking_history (int: packs per year OR category 0–3)
- symptoms: list of symptoms (string)
"""

import numpy as np


# Define a fixed symptom vocabulary
SYMPTOM_VOCAB = [
    "cough",
    "blood_cough",
    "chest_pain",
    "short_breath",
    "weight_loss",
    "back_pain",
    "fatigue",
    "hoarseness"
]

def encode_symptoms(symptoms):
    """
    symptoms = ["cough", "chest_pain"]
    returns one-hot vector of length len(SYMPTOM_VOCAB)
    """
    vec = np.zeros(len(SYMPTOM_VOCAB), dtype=np.float32)
    for i, s in enumerate(SYMPTOM_VOCAB):
        if s in symptoms:
            vec[i] = 1.0
    return vec


def encode_sex(sex_str):
    """
    Returns one-hot [male, female, other]
    """
    sex_str = sex_str.lower().strip()
    vec = np.zeros(3, dtype=np.float32)
    if sex_str == "male":
        vec[0] = 1
    elif sex_str == "female":
        vec[1] = 1
    else:
        vec[2] = 1
    return vec


def preprocess_metadata(raw_data):
    """
    raw_data example:
    {
        "age": 45,
        "sex": "male",
        "smoking_history": 2,
        "symptoms": ["cough", "short_breath"]
    }
    Returns clean numeric feature vector.
    """

    age = float(raw_data.get("age", 0)) / 100.0     # normalize age 0–1
    sex_vec = encode_sex(raw_data.get("sex", "other"))
    smoking = float(raw_data.get("smoking_history", 0)) / 3.0  # normalize category
    sym_vec = encode_symptoms(raw_data.get("symptoms", []))

    # Final combined vector
    features = np.concatenate([
        np.array([age], dtype=np.float32),
        sex_vec,
        np.array([smoking], dtype=np.float32),
        sym_vec
    ])

    return features
