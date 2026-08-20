from pathlib import Path
import pandas as pd
import numpy as np
from ..evaluation.metrics import haversine_km

REQUIRED=["facility_id","name","latitude","longitude","facility_type"]

def load_facilities(path):
    p=Path(path)
    if not p.exists(): return pd.DataFrame(columns=REQUIRED)
    df=pd.read_csv(p)
    missing=[c for c in REQUIRED if c not in df.columns]
    if missing: raise ValueError(f"Facility inventory missing columns: {missing}")
    return df

def rank_facilities(df,source_lat,source_lon,u10=None,v10=None,max_distance_km=20.0):
    rows=[]
    speed=np.hypot(u10 or 0,v10 or 0)
    for _,r in df.iterrows():
        d=haversine_km(source_lat,source_lon,float(r.latitude),float(r.longitude))
        if d>max_distance_km: continue
        # Predicted source itself should be close to facility. Wind is used only as
        # a secondary consistency cue: vector from facility -> inferred source should
        # not be strongly downwind if source inference is uncertain.
        dist_score=np.exp(-d/3.0)
        type_prior=1.0 if str(r.facility_type).lower() in {"oil","gas","oil_gas","landfill","coal","compressor"} else 0.7
        score=0.8*dist_score+0.2*type_prior
        rows.append({**r.to_dict(),"distance_km":d,"score":score})
    return pd.DataFrame(rows).sort_values("score",ascending=False).reset_index(drop=True) if rows else pd.DataFrame()
