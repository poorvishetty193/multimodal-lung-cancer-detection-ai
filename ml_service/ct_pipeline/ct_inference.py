# ct_inference.py
"""
Full CT inference pipeline:
1. Preprocess CT
2. Run UNet segmentation
3. Run 3D classifier
4. Produce cancer-risk score and segmented output
"""

import torch
import numpy as np

from ml_service.ct_pipeline.preprocess_ct import preprocess_ct
from ml_service.ct_pipeline.ct_unet_model import UNet
from ml_service.ct_pipeline.ct_classifier import CT3DClassifier

# -------------------------------------------
# Load Models
# -------------------------------------------

def load_unet_model(weight_path=None, device="cpu"):
    print("[INFO] Loading UNet segmentation model...")
    model = UNet(n_channels=1, n_classes=1).to(device)
    model.eval()

    if weight_path:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("[INFO] Loaded UNet weights.")
    else:
        print("[WARNING] No UNet weights provided — using untrained model.")

    return model


def load_classifier(weight_path=None, device="cpu"):
    print("[INFO] Loading 3D CT classifier...")
    model = CT3DClassifier().to(device)
    model.eval()

    if weight_path:
        model.load_state_dict(torch.load(weight_path, map_location=device))
        print("[INFO] Loaded classifier weights.")
    else:
        print("[WARNING] No classifier weights provided — using untrained model.")

    return model


# -------------------------------------------
# Inference Functions
# -------------------------------------------

def run_unet_segmentation(model, ct_slices, device="cpu"):
    """
    Input: slices → shape (Z, H, W)
    Output: predicted masks → shape (Z, H, W)
    """
    print("[INFO] Running UNet segmentation...")

    ct_slices = ct_slices.astype(np.float32)

    masks = []
    with torch.no_grad():
        for i in range(ct_slices.shape[0]):
            slice_tensor = torch.tensor(ct_slices[i]).unsqueeze(0).unsqueeze(0).to(device)
            pred = model(slice_tensor)
            pred_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8)
            masks.append(pred_mask)

    return np.stack(masks, axis=0)


def run_3d_classification(model, ct_volume, device="cpu"):
    """
    Input volume shape: (Z, H, W), normalized
    Output: cancer risk score
    """
    print("[INFO] Running 3D CT classifier...")

    volume = ct_volume.astype(np.float32)
    volume_tensor = torch.tensor(volume).unsqueeze(0).unsqueeze(0).to(device)

    with torch.no_grad():
        prob = model(volume_tensor).item()

    return float(prob)


# -------------------------------------------
# Master Pipeline
# -------------------------------------------

def ct_full_inference(path, is_dicom=True,
                      unet_weights=None,
                      classifier_weights=None,
                      device="cpu"):
    """
    Runs the full CT pipeline and returns:
    {
        "cancer_risk": float,
        "lung_mask_pred": np.ndarray,
        "preprocessed_ct": np.ndarray,
        "spacing": tuple
    }
    """

    print("\n==============================")
    print("     CT FULL INFERENCE")
    print("==============================\n")

    # Step 1 — Preprocess
    data = preprocess_ct(path, is_dicom=is_dicom)
    ct_volume = data["volume_preprocessed"]     # normalized (Z,H,W)
    spacing = data["spacing"]

    # Step 2 — Load models
    unet_model = load_unet_model(unet_weights, device)
    classifier_model = load_classifier(classifier_weights, device)

    # Step 3 — Segmentation
    lung_mask_pred = run_unet_segmentation(unet_model, ct_volume, device)

    # Step 4 — 3D classification
    cancer_prob = run_3d_classification(classifier_model, ct_volume, device)

    # Response
    return {
        "cancer_risk": cancer_prob,
        "lung_mask_pred": lung_mask_pred,
        "preprocessed_ct": ct_volume,
        "spacing": spacing
    }


# ---------------------------------------------------
# Quick Test
# ---------------------------------------------------

if __name__ == "__main__":
    print("Running CT inference test...")

    result = ct_full_inference(
        path="sample_ct_folder",
        is_dicom=True,
        unet_weights=None,
        classifier_weights=None,
        device="cpu"
    )

    print("\nCancer Risk Score:", result["cancer_risk"])
    print("CT Volume Shape:", result["preprocessed_ct"].shape)
    print("Mask Shape:", result["lung_mask_pred"].shape)
