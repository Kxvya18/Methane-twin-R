from pathlib import Path
import os
from ..utils import ensure_dirs

def _earthaccess_login():
    try:
        import earthaccess
    except ImportError as e:
        raise RuntimeError('Install research dependencies: pip install -e ".[research]"') from e

    # earthaccess supports .netrc, environment credentials, interactive login.
    try:
        auth = earthaccess.login(strategy="environment", persist=False)
        if not auth.authenticated:
            auth = earthaccess.login(strategy="netrc", persist=False)
        if not auth.authenticated:
            auth = earthaccess.login(persist=True)
        return earthaccess
    except Exception as e:
        raise RuntimeError(
            "NASA Earthdata authentication failed. Configure ~/.netrc or "
            "EARTHDATA_USERNAME/EARTHDATA_PASSWORD."
        ) from e

def download_emit_collection(short_name, version, start, end, bbox, out_dir, max_granules=None):
    earthaccess = _earthaccess_login()
    ensure_dirs(out_dir)
    results = earthaccess.search_data(
        short_name=short_name,
        version=version,
        temporal=(start, end),
        bounding_box=tuple(bbox),
    )
    if max_granules:
        results = results[: int(max_granules)]
    if not results:
        return {"collection": short_name, "found": 0, "downloaded_to": str(out_dir)}
    paths = earthaccess.download(results, local_path=str(out_dir))
    return {
        "collection": short_name,
        "found": len(results),
        "downloaded": len(paths),
        "downloaded_to": str(Path(out_dir).resolve()),
    }

def download_emit_from_config(cfg):
    e = cfg["emit"]
    p = cfg["paths"]
    plume = download_emit_collection(
        "EMITL2BCH4PLM", e.get("version","002"), e["start"], e["end"], e["bbox"],
        p["raw_plume"], e.get("max_plume_granules")
    )
    enh = download_emit_collection(
        "EMITL2BCH4ENH", e.get("version","002"), e["start"], e["end"], e["bbox"],
        p["raw_enhancement"], e.get("max_enhancement_granules")
    )
    return {"plume": plume, "enhancement": enh}
