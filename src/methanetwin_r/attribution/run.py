from pathlib import Path
import numpy as np,pandas as pd
from ..config import load_config
from ..utils import write_json
from ..evaluation.inference import predict_manifest
from .facilities import load_facilities,rank_facilities

def _norm_to_geo(x,y,bounds):
    left,bottom,right,top=bounds
    return top-y*(top-bottom), left+x*(right-left)

def run_attribution(cfg):
    facilities=load_facilities(cfg["paths"]["facilities"])
    if len(facilities)==0:
        raise FileNotFoundError(
            f"No facility inventory at {cfg['paths']['facilities']}. "
            "Provide CSV columns facility_id,name,latitude,longitude,facility_type."
        )
    pred,_=predict_manifest(cfg,split="test")
    allrows=[]
    for _,r in pred.iterrows():
        if r.detection_prob<0.5: continue
        z=np.load(r.npz_path,allow_pickle=True)
        lat,lon=_norm_to_geo(r.source_x_pred,r.source_y_pred,z["bounds"])
        u=float(z["u10"]) if np.isfinite(z["u10"]) else 0
        v=float(z["v10"]) if np.isfinite(z["v10"]) else 0
        ranked=rank_facilities(facilities,lat,lon,u,v,float(cfg["attribution"]["max_distance_km"]))
        for rank,(_,fr) in enumerate(ranked.head(10).iterrows(),start=1):
            allrows.append({"sample_id":r.sample_id,"rank":rank,"pred_source_lat":lat,"pred_source_lon":lon,**fr.to_dict()})
    out=pd.DataFrame(allrows)
    p=Path(cfg["paths"]["artifacts"])/"attribution"/"ranked_facilities.parquet"
    p.parent.mkdir(parents=True,exist_ok=True);out.to_parquet(p,index=False)
    report={"samples_ranked":int(out.sample_id.nunique()) if len(out) else 0,"rows":len(out),"output":str(p),
            "warning":"Facility ranking is a hypothesis, not proof of operational/legal responsibility."}
    write_json(p.with_suffix(".json"),report)
    return report
