from pathlib import Path
import numpy as np,pandas as pd
from sklearn.cluster import DBSCAN
from ..data.catalog import parse_timestamp_from_name
from ..data.raster import read_raster
from ..utils import write_json
from .persistence import beta_persistence

def _cluster_sources(events,eps_km=2.0):
    e=events.dropna(subset=["centroid_lat","centroid_lon"]).copy()
    if len(e)==0:return e
    coords=np.deg2rad(e[["centroid_lat","centroid_lon"]].to_numpy())
    labels=DBSCAN(eps=eps_km/6371.0088,min_samples=1,metric="haversine").fit_predict(coords)
    e["source_cluster"]=labels
    return e

def _enhancement_coverage(cfg):
    rows=[]
    for tif in Path(cfg["paths"]["raw_enhancement"]).rglob("*.tif"):
        if "CH4ENH" not in tif.name or "UNCERT" in tif.name or "SENS" in tif.name: continue
        ras=read_raster(tif)
        rows.append({"timestamp":parse_timestamp_from_name(tif.name),"bounds":ras["bounds"],"file":str(tif)})
    return rows

def run_persistence(cfg):
    ep=Path(cfg["paths"]["interim"])/"emit_events.parquet"
    if not ep.exists():raise FileNotFoundError("Build event catalog first.")
    events=_cluster_sources(pd.read_parquet(ep))
    scenes=_enhancement_coverage(cfg)
    results=[]
    for cluster,g in events.groupby("source_cluster"):
        lat=float(g.centroid_lat.mean());lon=float(g.centroid_lon.mean())
        detections=len(g)
        opportunities=0
        for s in scenes:
            left,bottom,right,top=s["bounds"]
            if left<=lon<=right and bottom<=lat<=top: opportunities+=1
        # At minimum each detected plume is an observation opportunity.
        opportunities=max(opportunities,detections)
        post=beta_persistence(detections,opportunities)
        results.append({"source_cluster":int(cluster),"latitude":lat,"longitude":lon,
                        "detections":detections,"opportunities":opportunities,**post})
    out=pd.DataFrame(results)
    p=Path(cfg["paths"]["artifacts"])/"persistence"/"posterior.parquet"
    p.parent.mkdir(parents=True,exist_ok=True);out.to_parquet(p,index=False)
    report={"clusters":len(out),"output":str(p),
            "note":"Opportunity count is based on downloaded enhancement-scene spatial coverage, not all possible EMIT overpasses."}
    write_json(p.with_suffix(".json"),report)
    return report
