# services/ml_service_ct/service/train.py
import torch, os
from torch.utils.data import DataLoader
from models import UNet3D
from dataset import CTDataset

def train(records, epochs=30, batch_size=2, out_dir="checkpoints"):
    ds = CTDataset(records)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=4)
    model = UNet3D(in_ch=1).cuda()
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    bce = torch.nn.BCELoss()
    ce = torch.nn.CrossEntropyLoss()
    for ep in range(epochs):
        model.train()
        for x, y in dl:
            x = x.cuda()
            out = model(x)
            seg = out["seg"]
            # use available segmentation mask if present (y.get("mask"))
            loss_seg = torch.tensor(0.).cuda()
            if y and "mask" in y:
                mask = torch.tensor(y["mask"]).unsqueeze(1).float().cuda()
                loss_seg = bce(seg, mask)
            # classification loss if label present
            loss_cls = torch.tensor(0.).cuda()
            if y and "label" in y:
                lbl = torch.tensor(y["label"]).long().cuda()
                loss_cls = ce(out["class_probs"], lbl)
            loss = loss_seg + 0.5*loss_cls
            opt.zero_grad()
            loss.backward()
            opt.step()
        # save checkpoint
        os.makedirs(out_dir, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(out_dir, f"model_ep{ep}.pt"))
