from pathlib import Path
import numpy as np
from ..utils import read_json
from ..models.network import build_model
from ..models.pipeline import _resolved_model_and_training
from ..utils import device_name

class Predictor:
    def __init__(self,cfg,checkpoint=None):
        import torch
        self.cfg=cfg
        self.device=torch.device(device_name())
        self.norm=read_json(Path(cfg["paths"]["processed"])/"normalization.json")
        model_cfg,_=_resolved_model_and_training(cfg)
        self.model=build_model(**model_cfg).to(self.device)
        ck=Path(checkpoint or (Path(cfg["paths"]["artifacts"])/"models"/"methanetwin_final.pt"))
        self.model.load_state_dict(torch.load(ck,map_location=self.device)["state_dict"])
        self.model.eval()
        cp=Path(cfg["paths"]["artifacts"])/"calibration.json"
        self.cal=read_json(cp) if cp.exists() else {"temperature":1.0,"conformal_log_q":None}

    def tensorize(self,img,valid,u10,v10):
        import torch
        e=np.log1p(np.maximum(img,0))/max(self.norm["log_scale_p995"],1e-6)
        e=np.clip(e,0,4)
        ws=self.norm.get("wind_scale",15)
        x=np.stack([e,np.full_like(e,u10/ws),np.full_like(e,v10/ws),valid],0).astype(np.float32)
        return torch.tensor(x[None],device=self.device)

    def predict(self,img,valid,u10,v10):
        import torch
        with torch.no_grad():
            o=self.model(self.tensorize(img,valid,u10,v10))
            logit=float(o["detection_logit"][0].cpu())/self.cal.get("temperature",1.0)
            p=1/(1+np.exp(-logit))
            q=float(torch.exp(o["log_q"][0]).cpu())
            src=o["source_xy"][0].cpu().numpy()
            mask=torch.sigmoid(o["mask_logit"])[0,0].cpu().numpy()
        cq=self.cal.get("conformal_log_q")
        interval=[q*np.exp(-cq),q*np.exp(cq)] if cq is not None else None
        return {"detection_probability":p,"emission_rate_kg_h":q,"emission_interval_kg_h":interval,
                "source_xy":[float(src[0]),float(src[1])],"mask":mask}
