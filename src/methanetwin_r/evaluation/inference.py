from pathlib import Path
import numpy as np,pandas as pd
from ..utils import read_json,device_name
from ..models.data import NPZDataset
from ..models.network import build_model
from ..models.pipeline import _resolved_model_and_training

def load_final_model(cfg, checkpoint=None):
    import torch
    ck=Path(checkpoint or (Path(cfg["paths"]["artifacts"])/"models"/"methanetwin_final.pt"))
    model_cfg,_=_resolved_model_and_training(cfg)
    model=build_model(**model_cfg)
    model.load_state_dict(torch.load(ck,map_location="cpu")["state_dict"])
    model.to(torch.device(device_name())).eval()
    return model

def predict_manifest(cfg, split="test", checkpoint=None):
    import torch
    mp=Path(cfg["paths"]["processed"])/"manifest.parquet"
    m=pd.read_parquet(mp); m=m[m.split==split].reset_index(drop=True)
    norm=read_json(Path(cfg["paths"]["processed"])/"normalization.json")
    ds=NPZDataset(m,norm,augment=False); model=load_final_model(cfg,checkpoint)
    rows=[]; masks=[]
    with torch.no_grad():
        for i in range(len(ds)):
            b=ds[i]; x=b["x"][None].to(next(model.parameters()).device)
            o=model(x)
            det=float(torch.sigmoid(o["detection_logit"])[0].cpu())
            mask=torch.sigmoid(o["mask_logit"])[0,0].cpu().numpy()
            src=o["source_xy"][0].cpu().numpy()
            q=float(torch.exp(o["log_q"][0]).cpu())
            sigma_log=float(torch.exp(0.5*o["logvar_q"][0]).cpu())
            r=m.iloc[i]
            rows.append({"sample_id":r.sample_id,"detection_true":int(r.detection),"detection_prob":det,
                         "q_true":float(r.emission_rate_kg_h) if pd.notna(r.emission_rate_kg_h) else np.nan,
                         "q_pred":q,"q_log_sigma":sigma_log,
                         "source_x_pred":float(src[0]),"source_y_pred":float(src[1]),
                         "source_label_quality":r.source_label_quality,"npz_path":r.npz_path})
            masks.append(mask)
    return pd.DataFrame(rows),masks
