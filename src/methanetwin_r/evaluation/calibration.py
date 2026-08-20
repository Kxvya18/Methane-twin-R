from pathlib import Path
import numpy as np
from scipy.optimize import minimize_scalar
from ..utils import write_json,read_json
from .inference import predict_manifest

def _nll_temperature(logits,y,T):
    z=logits/max(T,1e-4)
    p=1/(1+np.exp(-z))
    return -np.mean(y*np.log(p+1e-8)+(1-y)*np.log(1-p+1e-8))

def calibrate(cfg):
    pred,_=predict_manifest(cfg,split="calibration")
    y=pred.detection_true.to_numpy()
    p=np.clip(pred.detection_prob.to_numpy(),1e-6,1-1e-6)
    logits=np.log(p/(1-p))
    res=minimize_scalar(lambda T:_nll_temperature(logits,y,T),bounds=(0.2,5),method="bounded")
    pos=pred[np.isfinite(pred.q_true)&(pred.q_true>0)]
    alpha=float(cfg["calibration"]["alpha"])
    if len(pos):
        scores=np.abs(np.log(pos.q_true)-np.log(pos.q_pred))
        q=float(np.quantile(scores,min(1,np.ceil((len(scores)+1)*(1-alpha))/len(scores)),method="higher"))
    else: q=None
    artifact={"temperature":float(res.x),"conformal_log_q":q,"alpha":alpha,"n_calibration":len(pred),
              "n_emission_calibration":len(pos)}
    out=Path(cfg["paths"]["artifacts"])/"calibration.json"; write_json(out,artifact)
    return artifact
