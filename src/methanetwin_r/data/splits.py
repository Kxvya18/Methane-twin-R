import hashlib
import numpy as np
import pandas as pd

def region_id(lat, lon, bin_deg=2.0):
    return f"{int(np.floor((float(lat)+90)/bin_deg))}_{int(np.floor((float(lon)+180)/bin_deg))}"

def assign_group_splits(df, bin_deg=2.0):
    out = df.copy()
    out["region_id"] = [region_id(a,b,bin_deg) for a,b in zip(out["centroid_lat"],out["centroid_lon"])]
    groups = sorted(out["region_id"].dropna().unique())
    if len(groups) < 4:
        # deterministic per-sample fallback, explicitly marked
        def sample_split(s):
            x = int(hashlib.sha256(str(s).encode()).hexdigest()[:8],16)/0xffffffff
            return "train" if x<0.65 else "validation" if x<0.80 else "calibration" if x<0.90 else "test"
        out["split"] = out["sample_id"].map(sample_split)
        out["split_method"] = "sample_hash_fallback"
        return out

    def group_split(g):
        x = int(hashlib.sha256(str(g).encode()).hexdigest()[:8],16)/0xffffffff
        return "train" if x<0.65 else "validation" if x<0.80 else "calibration" if x<0.90 else "test"
    out["split"] = out["region_id"].map(group_split)
    out["split_method"] = "geographic_region_hash"
    return out
