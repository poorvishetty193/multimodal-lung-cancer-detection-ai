# services/ml_service_ct/service/infer.py
import torch, numpy as np
from models import UNet3D
from skimage.measure import label, regionprops
import json

def load_model(path):
    m = UNet3D(in_ch=1)
    m.load_state_dict(torch.load(path, map_location='cpu'))
    m.eval()
    return m

def postprocess_seg(seg_map, threshold=0.5, min_volume=10):
    seg_bin = seg_map > threshold
    L = label(seg_bin)
    props = regionprops(L)
    nodules = []
    for p in props:
        vol_vox = p.area
        if vol_vox < min_volume: continue
        z, y, x = map(int, p.centroid)
        # approximate diameter in mm: needs voxel spacing, use placeholder 5mm
        diameter_mm = ( (vol_vox)**(1/3) ) * 5.0
        nodules.append({"x": int(x), "y": int(y), "z": int(z), "diameter_mm": float(diameter_mm), "confidence": float(p.mean_intensity)})
    return nodules

def predict(model, volume_tensor):
    with torch.no_grad():
        out = model(volume_tensor.unsqueeze(0))
    seg = out["seg"].squeeze().cpu().numpy()
    nodules = postprocess_seg(seg)
    embedding = out["embedding"].squeeze().cpu().numpy().tolist()
    lung_summary = {"volume_cc": float((volume_tensor.numpy()>0).sum())} 
    return {"job_id": None, "nodules": nodules, "lung_mask_summary": lung_summary, "ct_embedding": embedding}
