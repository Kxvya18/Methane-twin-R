from pathlib import Path
import json, re
import numpy as np

def flatten_dict(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k,v in obj.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(flatten_dict(v, key))
    elif isinstance(obj, list):
        for i,v in enumerate(obj):
            out.update(flatten_dict(v, f"{prefix}[{i}]"))
    else:
        out[prefix] = obj
    return out

def _to_float(v):
    if isinstance(v, (int,float)) and np.isfinite(v):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", v.replace(",",""))
        if m:
            try: return float(m.group(0))
            except ValueError: pass
    return None

def _find_numeric(flat, include_terms, exclude_terms=()):
    candidates = []
    for k,v in flat.items():
        kl = k.lower()
        if all(term in kl for term in include_terms) and not any(term in kl for term in exclude_terms):
            x = _to_float(v)
            if x is not None: candidates.append((k,x))
    return candidates

def _units_context(flat, key):
    prefix=key.rsplit(".",1)[0] if "." in key else ""
    texts=[key]
    for k,v in flat.items():
        if prefix and not k.startswith(prefix): continue
        if any(t in k.lower() for t in ("unit","units")):
            texts.append(str(v))
    return " ".join(texts).lower().replace(" ", "")

def _emission_to_kg_h(flat, key, value):
    if key is None or value is None: return None, None
    ctx=_units_context(flat,key)
    # Safe, explicit unit conversion. If units are ambiguous we do not silently
    # manufacture a kg/h label.
    if ("kg" in ctx) and any(t in ctx for t in ("/h","/hr","hour-1","h-1","hr-1","perhour")):
        return float(value), "kg/h"
    if any(t in ctx for t in ("tonne/h","tonnes/h","t/h","ton/hr","tonneperhour")):
        return float(value)*1000.0, "kg/h_from_tonnes"
    # Some product schemas put kg/hr directly in the field name.
    kl=key.lower().replace(" ","")
    if any(t in kl for t in ("kg_hr","kg/hr","kghr","kg_h","kgh-1")):
        return float(value), "kg/h"
    return None, "ambiguous"

def _geometry_centroid(obj):
    try:
        from shapely.geometry import shape
        if obj.get("type") == "FeatureCollection":
            geoms = [shape(f["geometry"]) for f in obj.get("features",[]) if f.get("geometry")]
            if not geoms: return None
            from shapely.ops import unary_union
            c = unary_union(geoms).centroid
            return float(c.x), float(c.y)
        if obj.get("type") == "Feature" and obj.get("geometry"):
            c = shape(obj["geometry"]).centroid
            return float(c.x), float(c.y)
        if obj.get("type") in {"Polygon","MultiPolygon","Point"}:
            c = shape(obj).centroid
            return float(c.x), float(c.y)
    except Exception:
        return None
    return None

def parse_emit_plume_metadata(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    flat = flatten_dict(obj)

    emission_candidates = []
    for terms in [("emission","rate"),("emission",),("flux",)]:
        emission_candidates += _find_numeric(flat, terms, exclude_terms=("uncert","lower","upper","min","max"))
    emission_candidates = sorted(
        set(emission_candidates),
        key=lambda kv: (("kg" not in kv[0].lower()) + ("hr" not in kv[0].lower() and "hour" not in kv[0].lower()),len(kv[0]))
    )
    raw_emission = emission_candidates[0] if emission_candidates else (None,None)
    emission_kg_h, unit_status = _emission_to_kg_h(flat, raw_emission[0], raw_emission[1])

    unc_candidates = _find_numeric(flat, ("emission","uncert"))
    if not unc_candidates: unc_candidates = _find_numeric(flat, ("uncert",))
    emission_unc = unc_candidates[0] if unc_candidates else (None,None)

    lat_cands = _find_numeric(flat, ("source","lat")) + _find_numeric(flat, ("source","latitude"))
    lon_cands = _find_numeric(flat, ("source","lon")) + _find_numeric(flat, ("source","longitude"))
    source_lat = lat_cands[0] if lat_cands else (None,None)
    source_lon = lon_cands[0] if lon_cands else (None,None)

    centroid = _geometry_centroid(obj)
    if centroid is None:
        for f in obj.get("features", []) if isinstance(obj,dict) else []:
            centroid = _geometry_centroid(f)
            if centroid is not None: break

    return {
        "emission_rate_value": emission_kg_h,
        "emission_rate_key": raw_emission[0],
        "emission_rate_raw_value": raw_emission[1],
        "emission_unit_status": unit_status,
        "emission_uncertainty_value": emission_unc[1],
        "emission_uncertainty_key": emission_unc[0],
        "source_latitude": source_lat[1],
        "source_latitude_key": source_lat[0],
        "source_longitude": source_lon[1],
        "source_longitude_key": source_lon[0],
        "geometry_centroid_lon": centroid[0] if centroid else None,
        "geometry_centroid_lat": centroid[1] if centroid else None,
        "metadata_path": str(path),
    }

def candidate_metadata_fields(path):
    obj = json.loads(Path(path).read_text(encoding="utf-8"))
    flat = flatten_dict(obj)
    return {k:v for k,v in flat.items() if any(t in k.lower() for t in ("emission","flux","source","uncert","lat","lon","unit"))}
