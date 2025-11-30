# services/ml_service_ct/service/dataset.py
import os, numpy as np
import torch
from torch.utils.data import Dataset
import nibabel as nib
import pydicom
import zipfile
import io
from skimage.transform import resize
from PIL import Image

def load_nifti(path):
    img = nib.load(path)
    data = img.get_fdata()
    return data.astype(np.float32)

def load_image_stack(path):
    # try reading PNG/JPG stack single file (assume single image)
    im = Image.open(path)
    arr = np.array(im).astype(np.float32)
    if arr.ndim == 2: arr = arr[np.newaxis,...]
    return arr

class CTDataset(Dataset):
    def __init__(self, records, target_shape=(128,128,128), transforms=None):
        self.records = records
        self.shape = target_shape
        self.transforms = transforms
    def __len__(self): return len(self.records)
    def __getitem__(self, idx):
        rec = self.records[idx]
        path = rec["ct_path"]
        if path.endswith(('.nii', '.nii.gz')):
            vol = load_nifti(path)
        else:
            vol = load_image_stack(path)
        # resample/resize to target
        vol = resize(vol, self.shape, preserve_range=True, anti_aliasing=True)
        if vol.ndim==3:
            # ensure channels: C x D x H x W
            arr = vol[np.newaxis,...]
        else:
            arr = vol
        arr = (arr - arr.mean()) / (arr.std()+1e-8)
        return torch.tensor(arr, dtype=torch.float32), rec.get("label", {})
