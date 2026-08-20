from pathlib import Path
import re
import pandas as pd
from .metadata import parse_emit_plume_metadata
from .raster import read_raster
from ..utils import ensure_dirs, sha256_file, write_json

TS_RE = re.compile(r"(\d{8}T\d{6})")
ID_RE = re.compile(r"_(\d{6})(?:\.|_)")

def parse_timestamp_from_name(name):
    m = TS_RE.search(name)
    if not m:
        return None
    return pd.to_datetime(m.group(1), format="%Y%m%dT%H%M%S", utc=True)

def _find_metadata_for_tif(tif):
    stem = tif.stem
    # Same base name expected for PLM product.
    for ext in (".json",".geojson"):
        p = tif.with_suffix(ext)
        if p.exists():
            return p
    # fallback same timestamp/id
    candidates = list(tif.parent.glob("*.json")) + list(tif.parent.glob("*.geojson"))
    for p in candidates:
        if stem in p.stem or p.stem in stem:
            return p
    return None

def build_emit_catalog(cfg):
    raw = Path(cfg["paths"]["raw_plume"])
    interim = Path(cfg["paths"]["interim"])
    ensure_dirs(interim)
    rows = []
    for tif in sorted(raw.rglob("*.tif")):
        if "CH4PLM" not in tif.name:
            continue
        meta_path = _find_metadata_for_tif(tif)
        md = parse_emit_plume_metadata(meta_path) if meta_path else {}
        ras = read_raster(tif)
        left,bottom,right,top = ras["bounds"]
        centroid_lon = md.get("geometry_centroid_lon")
        centroid_lat = md.get("geometry_centroid_lat")
        if centroid_lon is None:
            centroid_lon = (left+right)/2
        if centroid_lat is None:
            centroid_lat = (bottom+top)/2

        source_quality = "none"
        if md.get("source_latitude") is not None and md.get("source_longitude") is not None:
            source_quality = "metadata_exact"

        idm = ID_RE.search(tif.name)
        rows.append({
            "sample_id": tif.stem,
            "plume_id": idm.group(1) if idm else tif.stem,
            "timestamp": parse_timestamp_from_name(tif.name),
            "plume_tif": str(tif),
            "metadata_json": str(meta_path) if meta_path else None,
            "sha256": sha256_file(tif),
            "left": left, "bottom": bottom, "right": right, "top": top,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "emission_rate_kg_h": md.get("emission_rate_value"),
            "emission_uncertainty": md.get("emission_uncertainty_value"),
            "emission_rate_key": md.get("emission_rate_key"),
            "emission_rate_raw_value": md.get("emission_rate_raw_value"),
            "emission_unit_status": md.get("emission_unit_status"),
            "source_lat": md.get("source_latitude"),
            "source_lon": md.get("source_longitude"),
            "source_label_quality": source_quality,
        })

    df = pd.DataFrame(rows)
    out = interim/"emit_events.parquet"
    if len(df):
        df.to_parquet(out, index=False)
    else:
        df.to_pickle(interim/"emit_events.empty.pkl")
    report = {
        "plume_files": len(df),
        "with_metadata": int(df["metadata_json"].notna().sum()) if len(df) else 0,
        "with_emission_label": int(df["emission_rate_kg_h"].notna().sum()) if len(df) else 0,
        "with_exact_source_label": int((df["source_label_quality"]=="metadata_exact").sum()) if len(df) else 0,
        "output": str(out),
    }
    write_json(interim/"catalog_report.json", report)
    return report

def metadata_report(cfg):
    p = Path(cfg["paths"]["interim"])/"emit_events.parquet"
    if not p.exists():
        raise FileNotFoundError("Run `methanetwin data catalog` first.")
    df = pd.read_parquet(p)
    return {
        "n": len(df),
        "emission_label_rate": float(df["emission_rate_kg_h"].notna().mean()) if len(df) else 0,
        "source_exact_rate": float((df["source_label_quality"]=="metadata_exact").mean()) if len(df) else 0,
        "emission_keys": df["emission_rate_key"].value_counts(dropna=False).head(20).to_dict(),
        "emission_unit_status": df["emission_unit_status"].value_counts(dropna=False).to_dict() if "emission_unit_status" in df else {},
        "emission_min": float(df["emission_rate_kg_h"].min()) if df["emission_rate_kg_h"].notna().any() else None,
        "emission_max": float(df["emission_rate_kg_h"].max()) if df["emission_rate_kg_h"].notna().any() else None,
    }
