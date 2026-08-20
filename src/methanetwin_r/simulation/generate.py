from pathlib import Path
import numpy as np
import pandas as pd
from ..physics.plume import gaussian_plume_2d
from ..utils import ensure_dirs, write_json, set_seed

def generate_physics_dataset(cfg):
    set_seed(int(cfg["seed"]))
    pcfg=cfg["physics"]
    n=int(pcfg["synthetic_samples"]); size=int(pcfg["image_size"])
    outdir=Path(cfg["paths"]["processed"])/"synthetic"
    ensure_dirs(outdir)
    rng=np.random.default_rng(int(cfg["seed"]))
    rows=[]
    for i in range(n):
        positive=rng.random()>0.18
        if positive:
            q=float(np.exp(rng.uniform(np.log(pcfg["q_min_kg_h"]),np.log(pcfg["q_max_kg_h"]))))
            speed=float(rng.uniform(pcfg["wind_min_m_s"],pcfg["wind_max_m_s"]))
            direction=float(rng.uniform(0,360))
            source=(float(rng.uniform(0.12,0.55)),float(rng.uniform(0.15,0.85)))
            d=gaussian_plume_2d(size=size,pixel_size_m=pcfg["pixel_size_m"],source_xy=source,
                                q_kg_h=q,wind_speed=speed,wind_dir_deg=direction,
                                dispersion=float(rng.uniform(0.6,1.8)),
                                noise=float(rng.uniform(0.01,0.15)),
                                cloud_fraction=float(rng.uniform(0,0.20)),
                                seed=int(rng.integers(0,2**31-1)))
            detection=1
        else:
            speed=float(rng.uniform(pcfg["wind_min_m_s"],pcfg["wind_max_m_s"]))
            direction=float(rng.uniform(0,360))
            u=speed*np.cos(np.deg2rad(direction)); v=speed*np.sin(np.deg2rad(direction))
            bg=rng.normal(0,0.02,(size,size)).astype(np.float32)
            bg=np.maximum(bg,0)
            d={"enhancement":bg,"mask":np.zeros_like(bg),"valid":np.ones_like(bg),
               "u10":u,"v10":v,"source_xy":np.array([np.nan,np.nan],dtype=np.float32),"q_kg_h":np.nan}
            detection=0
        p=outdir/f"SYN_{i:07d}.npz"
        np.savez_compressed(p,enhancement=d["enhancement"],plume_mask=d["mask"],valid=d["valid"],
                            u10=np.float32(d["u10"]),v10=np.float32(d["v10"]),
                            source_xy=d["source_xy"],emission_rate=np.float32(d["q_kg_h"]),
                            detection=np.float32(detection),source_quality=np.asarray("exact_synthetic"))
        rows.append({"sample_id":p.stem,"npz_path":str(p),"detection":detection,
                     "emission_rate_kg_h":d["q_kg_h"],"source_label_quality":"exact_synthetic",
                     "split":"train" if i<int(n*0.9) else "validation","label_origin":"PHYSICS_SIM"})
    m=pd.DataFrame(rows)
    mp=Path(cfg["paths"]["processed"])/"synthetic_manifest.parquet"
    m.to_parquet(mp,index=False)
    report={"samples":n,"positive":int(m.detection.sum()),"output":str(mp)}
    write_json(Path(cfg["paths"]["processed"])/"synthetic_report.json",report)
    return report
