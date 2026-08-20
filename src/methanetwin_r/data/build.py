from pathlib import Path
import json, math, random
import numpy as np
import pandas as pd
from .raster import read_raster, resize_array
from .splits import assign_group_splits
from ..utils import ensure_dirs, sha256_file, write_json, set_seed

def _upwind_proxy_source(img, valid, u, v):
    # Proxy only: find the upwind-most high-enhancement pixel.
    pos = np.maximum(img,0)
    threshold = np.nanpercentile(pos[valid>0], 85) if (valid>0).any() else 0
    ys,xs = np.where((pos>=threshold) & (valid>0))
    if len(xs)==0:
        return None
    # Image x right=east, y down=south. Wind vector u=east, v=north.
    # Projection of pixel coordinate along wind direction; source is most upwind -> minimum.
    dx = xs - img.shape[1]/2
    dy_north = -(ys - img.shape[0]/2)
    norm = max(math.hypot(u,v),1e-6)
    proj = dx*(u/norm) + dy_north*(v/norm)
    j = int(np.argmin(proj))
    return float(xs[j]/max(1,img.shape[1]-1)), float(ys[j]/max(1,img.shape[0]-1))

def _save_npz(path, enhancement, valid, mask, u10, v10, q, source_xy, source_quality, detection, bounds):
    np.savez_compressed(
        path,
        enhancement=enhancement.astype("float32"),
        valid=valid.astype("float32"),
        plume_mask=mask.astype("float32"),
        u10=np.float32(u10 if u10 is not None else np.nan),
        v10=np.float32(v10 if v10 is not None else np.nan),
        emission_rate=np.float32(q if q is not None else np.nan),
        source_xy=np.asarray(source_xy if source_xy is not None else [np.nan,np.nan], dtype="float32"),
        source_quality=np.asarray(source_quality),
        detection=np.float32(detection),
        bounds=np.asarray(bounds,dtype="float64"),
    )

def _real_positive_samples(cfg, events, era5, sample_dir):
    size = int(cfg["dataset"]["image_size"])
    minpix = int(cfg["dataset"]["min_positive_pixels"])
    merged = events.merge(era5, on="sample_id", how="left") if len(era5) else events.copy()
    rows = []
    for _,r in merged.iterrows():
        ras = read_raster(r["plume_tif"])
        img, valid = resize_array(ras["array"], size, fill=0.0, order=1)
        plume_mask = ((img>0) & (valid>0)).astype(np.float32)
        if plume_mask.sum() < minpix:
            continue
        u,v = r.get("u10"), r.get("v10")
        source_xy = None
        source_quality = str(r.get("source_label_quality","none"))
        # Exact geographic source -> normalized raster coordinate
        if source_quality=="metadata_exact" and pd.notna(r.get("source_lon")) and pd.notna(r.get("source_lat")):
            left,bottom,right,top = ras["bounds"]
            sx = (float(r["source_lon"])-left)/max(right-left,1e-12)
            sy = (top-float(r["source_lat"]))/max(top-bottom,1e-12)
            source_xy = [float(np.clip(sx,0,1)), float(np.clip(sy,0,1))]
        elif pd.notna(u) and pd.notna(v):
            source_xy = _upwind_proxy_source(img,valid,float(u),float(v))
            source_quality = "physics_proxy" if source_xy else "none"

        out = sample_dir/f"{r['sample_id']}.npz"
        _save_npz(out,img,valid,plume_mask,
                  float(u) if pd.notna(u) else None,
                  float(v) if pd.notna(v) else None,
                  float(r["emission_rate_kg_h"]) if pd.notna(r.get("emission_rate_kg_h")) else None,
                  source_xy,source_quality,1.0,ras["bounds"])
        rows.append({
            **{k:r.get(k) for k in ["sample_id","timestamp","centroid_lat","centroid_lon","emission_rate_kg_h",
                                    "source_lat","source_lon","source_label_quality"]},
            "source_label_quality":source_quality,
            "npz_path":str(out),
            "detection":1,
            "label_origin":"EMIT_CH4PLM_V002",
        })
    return rows

def _negative_samples(cfg, sample_dir):
    from .catalog import parse_timestamp_from_name
    raw = Path(cfg["paths"]["raw_enhancement"])
    size = int(cfg["dataset"]["image_size"])
    n_per = int(cfg["dataset"]["negatives_per_enhancement_scene"])
    max_thr = float(cfg["dataset"]["negative_max_ppmm"])
    rng = np.random.default_rng(int(cfg["seed"]))
    rows=[]
    for tif in sorted(raw.rglob("*.tif")):
        if "CH4ENH" not in tif.name or "UNCERT" in tif.name or "SENS" in tif.name:
            continue
        ras=read_raster(tif); a=ras["array"]
        # sample windows from original raster when possible
        h,w=a.shape
        win=max(16,min(h,w,int(min(h,w)*0.25)))
        successes=0
        for attempt in range(n_per*20):
            if successes>=n_per: break
            if h<=win or w<=win:
                patch=a
            else:
                top=int(rng.integers(0,h-win)); left=int(rng.integers(0,w-win))
                patch=a[top:top+win,left:left+win]
            finite=patch[np.isfinite(patch)]
            if finite.size<20: continue
            if np.nanmax(finite)>max_thr: continue
            img,valid=resize_array(patch,size,fill=0.0,order=1)
            sid=f"NEG_{tif.stem}_{successes:02d}"
            out=sample_dir/f"{sid}.npz"
            _save_npz(out,img,valid,np.zeros_like(img),np.nan,np.nan,None,None,"none",0.0,ras["bounds"])
            leftb,bottomb,rightb,topb=ras["bounds"]
            rows.append({
                "sample_id":sid,
                "timestamp":parse_timestamp_from_name(tif.name),
                "centroid_lat":(bottomb+topb)/2,
                "centroid_lon":(leftb+rightb)/2,
                "emission_rate_kg_h":np.nan,
                "source_lat":np.nan,"source_lon":np.nan,
                "source_label_quality":"none",
                "npz_path":str(out),
                "detection":0,
                "label_origin":"EMIT_CH4ENH_LOW_BACKGROUND_V002",
            })
            successes+=1
    return rows

def _fit_normalization(manifest):
    vals=[]
    for p in manifest.loc[manifest["split"]=="train","npz_path"]:
        z=np.load(p,allow_pickle=True)
        x=np.maximum(z["enhancement"],0)
        x=np.log1p(x)
        vals.append(x[z["valid"]>0])
    if not vals:
        return {"log_scale_p995":1.0,"wind_scale":15.0}
    v=np.concatenate(vals)
    scale=float(np.quantile(v,0.995))
    return {"log_scale_p995":max(scale,1e-6),"wind_scale":15.0}

def build_real_dataset(cfg):
    set_seed(int(cfg["seed"]))
    interim=Path(cfg["paths"]["interim"])
    processed=Path(cfg["paths"]["processed"])
    sample_dir=processed/"samples"
    ensure_dirs(processed,sample_dir)
    events_path=interim/"emit_events.parquet"
    if not events_path.exists():
        raise FileNotFoundError("Run `methanetwin data catalog` first.")
    events=pd.read_parquet(events_path)
    era_path=interim/"era5_events.parquet"
    era=pd.read_parquet(era_path) if era_path.exists() else pd.DataFrame(columns=["sample_id","u10","v10"])
    rows=_real_positive_samples(cfg,events,era,sample_dir)
    rows += _negative_samples(cfg,sample_dir)
    manifest=pd.DataFrame(rows)
    if len(manifest)==0:
        raise RuntimeError("No usable samples were created.")
    manifest=assign_group_splits(manifest,float(cfg["dataset"]["region_bin_deg"]))
    manifest_path=processed/"manifest.parquet"
    manifest.to_parquet(manifest_path,index=False)
    norm=_fit_normalization(manifest)
    write_json(processed/"normalization.json",norm)
    qc={
        "samples":len(manifest),
        "positives":int((manifest.detection==1).sum()),
        "negatives":int((manifest.detection==0).sum()),
        "emission_labeled":int(manifest.emission_rate_kg_h.notna().sum()),
        "exact_source_labeled":int((manifest.source_label_quality=="metadata_exact").sum()),
        "proxy_source_labeled":int((manifest.source_label_quality=="physics_proxy").sum()),
        "split_counts":manifest["split"].value_counts().to_dict(),
        "split_method":manifest["split_method"].value_counts().to_dict(),
    }
    write_json(processed/"quality_report.json",qc)
    provenance={
        "config":cfg["_config_path"],
        "manifest_sha256":sha256_file(manifest_path),
        "source_collections":["EMITL2BCH4PLM V002","EMITL2BCH4ENH V002"],
    }
    write_json(processed/"provenance.json",provenance)
    return qc
