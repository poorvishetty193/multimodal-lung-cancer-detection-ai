# preprocess_ct.py
"""
CT Preprocessing Pipeline
- Reads DICOM or NIfTI
- Converts to 3D volume
- Resamples to isotropic spacing
- Applies lung window
- Normalizes volume
"""

import os
import numpy as np
import SimpleITK as sitk
from skimage import morphology


# ---------------------------
# Load CT data
# ---------------------------

def load_dicom_series(folder_path):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(folder_path)
    reader.SetFileNames(dicom_names)
    image = reader.Execute()

    volume = sitk.GetArrayFromImage(image)  # (z, y, x)
    spacing = image.GetSpacing()            # (x, y, z)
    spacing = (spacing[2], spacing[1], spacing[0])  # reorder to (z,y,x)
    return volume.astype(np.int16), spacing


def load_nifti(path):
    img = sitk.ReadImage(path)
    volume = sitk.GetArrayFromImage(img)
    spacing = img.GetSpacing()
    spacing = (spacing[2], spacing[1], spacing[0])
    return volume.astype(np.int16), spacing


# ---------------------------
# Preprocessing Steps
# ---------------------------

def lung_window(volume, WL=-600, WW=1500):
    lower = WL - WW // 2
    upper = WL + WW // 2
    volume = np.clip(volume, lower, upper)
    return (volume - lower) / (upper - lower)


def normalize(volume):
    return (volume - volume.mean()) / (volume.std() + 1e-8)


def simple_lung_mask(volume_hu):
    mask = volume_hu < -400   # threshold for lung tissue
    mask = morphology.remove_small_objects(mask, min_size=500)
    mask = morphology.remove_small_holes(mask, area_threshold=500)
    return mask.astype(np.uint8)


# ---------------------------
# Master Preprocessing Function
# ---------------------------

def preprocess_ct(path, is_dicom=True):
    if is_dicom:
        vol, spacing = load_dicom_series(path)
    else:
        vol, spacing = load_nifti(path)

    lung_win = lung_window(vol)
    mask = simple_lung_mask(vol)
    lung_only = lung_win * mask
    norm = normalize(lung_only)

    return {
        "volume_raw": vol,
        "volume_preprocessed": norm,
        "lung_mask": mask,
        "spacing": spacing
    }


if __name__ == "__main__":
    print("Preprocessing CT example...")
