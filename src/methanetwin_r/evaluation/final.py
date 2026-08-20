from pathlib import Path
import numpy as np,pandas as pd
from ..utils import read_json,write_json
from .inference import predict_manifest
from .metrics import binary_metrics,mask_metrics,regression_metrics,haversine_km

def _norm_source_to_latlon(x,y,bounds):
    left,bottom,right,top=bounds
    lon=left+x*(right-left)
    lat=top-y*(top-bottom)
    return lat,lon

def evaluate_final(cfg):
    pred,masks=predict_manifest(cfg,split="test")
    calp=Path(cfg["paths"]["artifacts"])/"calibration.json"
    cal=read_json(calp) if calp.exists() else {"temperature":1.0,"conformal_log_q":None,"alpha":0.05}
    # recalibrate detection probabilities
    p=np.clip(pred.detection_prob.to_numpy(),1e-6,1-1e-6)
    z=np.log(p/(1-p))/cal["temperature"]; pcal=1/(1+np.exp(-z))
    result={"detection":binary_metrics(pred.detection_true,pcal)}

    mask_metrics_rows=[]
    source_exact=[]
    qtrue=[];qpred=[];covered=[];width=[]
    for i,r in pred.iterrows():
        znp=np.load(r.npz_path,allow_pickle=True)
        mm=mask_metrics(znp["plume_mask"],masks[i]); mask_metrics_rows.append(mm)
        if np.isfinite(r.q_true) and r.q_true>0:
            qtrue.append(r.q_true);qpred.append(r.q_pred)
            if cal.get("conformal_log_q") is not None:
                q=cal["conformal_log_q"]; lo=r.q_pred*np.exp(-q); hi=r.q_pred*np.exp(q)
                covered.append(lo<=r.q_true<=hi); width.append(hi-lo)
        if str(r.source_label_quality)=="metadata_exact" and np.isfinite(znp["source_xy"]).all():
            pred_lat,pred_lon=_norm_source_to_latlon(r.source_x_pred,r.source_y_pred,znp["bounds"])
            true_lat,true_lon=_norm_source_to_latlon(float(znp["source_xy"][0]),float(znp["source_xy"][1]),znp["bounds"])
            source_exact.append(haversine_km(true_lat,true_lon,pred_lat,pred_lon))

    result["segmentation"]={
        "mean_iou":float(np.mean([x["iou"] for x in mask_metrics_rows])) if mask_metrics_rows else None,
        "mean_dice":float(np.mean([x["dice"] for x in mask_metrics_rows])) if mask_metrics_rows else None,
    }
    result["emission"]=regression_metrics(qtrue,qpred) if qtrue else None
    result["uncertainty"]={
        "target_coverage":1-cal.get("alpha",0.05),
        "empirical_coverage":float(np.mean(covered)) if covered else None,
        "mean_interval_width_kg_h":float(np.mean(width)) if width else None,
    }
    result["real_exact_source_localization"]={
        "n":len(source_exact),
        "mean_km":float(np.mean(source_exact)) if source_exact else None,
        "median_km":float(np.median(source_exact)) if source_exact else None,
        "note":"Proxy source labels are excluded from this metric."
    }
    out=Path(cfg["paths"]["artifacts"])/"evaluation"/"final_test_metrics.json";write_json(out,result)
    pred.to_parquet(Path(cfg["paths"]["artifacts"])/"evaluation"/"test_predictions.parquet",index=False)
    return result
