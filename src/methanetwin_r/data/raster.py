from pathlib import Path
import numpy as np

def read_raster(path):
    try:
        import rasterio
    except ImportError as e:
        raise RuntimeError('Install research dependencies: pip install -e ".[research]"') from e
    with rasterio.open(path) as src:
        a = src.read(1).astype("float32")
        nodata = src.nodata
        if nodata is not None:
            a[a == nodata] = np.nan
        a[~np.isfinite(a)] = np.nan
        return {
            "array": a,
            "transform": src.transform,
            "crs": str(src.crs),
            "bounds": tuple(src.bounds),
            "width": src.width,
            "height": src.height,
        }

def resize_array(a, size, fill=0.0, order=1):
    from scipy.ndimage import zoom
    a = np.asarray(a, dtype=np.float32)
    valid = np.isfinite(a)
    if not valid.any():
        return np.full((size,size), fill, dtype=np.float32), np.zeros((size,size), dtype=np.float32)
    v = np.where(valid, a, fill)
    scale = (size/a.shape[0], size/a.shape[1])
    out = zoom(v, scale, order=order)
    m = zoom(valid.astype(np.float32), scale, order=0)
    # scipy may produce off-by-one from floating rounding
    out = out[:size,:size]
    m = m[:size,:size]
    if out.shape != (size,size):
        pad = np.full((size,size), fill, dtype=np.float32)
        pm = np.zeros((size,size), dtype=np.float32)
        pad[:out.shape[0],:out.shape[1]] = out
        pm[:m.shape[0],:m.shape[1]] = m
        out,m = pad,pm
    return out.astype(np.float32), m.astype(np.float32)

def crop_window(a, top, left, size):
    return a[top:top+size, left:left+size]
