from pathlib import Path
import time
import pandas as pd
import numpy as np
from ..utils import ensure_dirs, write_json

def _request_event(row, variables, out_file, half_box=0.25):
    import cdsapi
    t = pd.Timestamp(row["timestamp"]).tz_convert("UTC")
    lat = float(row["centroid_lat"])
    lon = float(row["centroid_lon"])
    request = {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [f"{t.year:04d}"],
        "month": [f"{t.month:02d}"],
        "day": [f"{t.day:02d}"],
        "time": [f"{t.hour:02d}:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": [lat+half_box, lon-half_box, lat-half_box, lon+half_box],
    }
    cdsapi.Client().retrieve("reanalysis-era5-single-levels", request, str(out_file))

def _extract_event(nc_path, lat, lon):
    import xarray as xr
    ds = xr.open_dataset(nc_path)
    # Coordinate names vary by backend/release.
    lat_name = "latitude" if "latitude" in ds.coords else "lat"
    lon_name = "longitude" if "longitude" in ds.coords else "lon"
    point = ds.sel({lat_name: lat, lon_name: lon}, method="nearest")
    def val(*names):
        for n in names:
            if n in point:
                return float(np.asarray(point[n]).squeeze())
        return None
    u = val("u10","10m_u_component_of_wind")
    v = val("v10","10m_v_component_of_wind")
    return {
        "u10": u,
        "v10": v,
        "wind_speed": float(np.hypot(u,v)) if u is not None and v is not None else None,
        "t2m": val("t2m","2m_temperature"),
        "surface_pressure": val("sp","surface_pressure"),
        "boundary_layer_height": val("blh","boundary_layer_height"),
    }

def download_era5_for_events(cfg):
    events_path = Path(cfg["paths"]["interim"])/"emit_events.parquet"
    if not events_path.exists():
        raise FileNotFoundError("Run `methanetwin data catalog` first.")
    df = pd.read_parquet(events_path)
    raw_dir = Path(cfg["paths"]["raw_era5"])
    ensure_dirs(raw_dir)
    rows = []
    variables = cfg["era5"]["variables"]
    half = float(cfg["era5"].get("half_box_deg",0.25))

    for i,row in df.iterrows():
        sid = row["sample_id"]
        out = raw_dir/f"{sid}.nc"
        try:
            if not out.exists():
                _request_event(row, variables, out, half_box=half)
                time.sleep(0.2)
            vals = _extract_event(out, float(row["centroid_lat"]), float(row["centroid_lon"]))
            rows.append({"sample_id":sid, **vals, "era5_file":str(out), "status":"ok"})
        except Exception as e:
            rows.append({"sample_id":sid, "status":"error", "error":repr(e)})
    outdf = pd.DataFrame(rows)
    outp = Path(cfg["paths"]["interim"])/"era5_events.parquet"
    outdf.to_parquet(outp,index=False)
    report = {"events":len(df),"success":int((outdf["status"]=="ok").sum()),"output":str(outp)}
    write_json(Path(cfg["paths"]["interim"])/"era5_report.json", report)
    return report
