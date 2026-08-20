import numpy as np
from scipy.ndimage import gaussian_filter

def gaussian_plume_2d(size=128, pixel_size_m=60.0, source_xy=(0.25,0.5),
                      q_kg_h=1000.0, wind_speed=5.0, wind_dir_deg=0.0,
                      dispersion=1.0, noise=0.03, cloud_fraction=0.0, seed=None):
    """
    Research simulation for supervised pretraining.

    wind_dir_deg: direction the plume travels toward in image/world coordinates,
    0=east/right, 90=north/up.
    The amplitude is proportional to Q / wind, while crosswind spread grows downwind.
    This is an intentionally simplified column-enhancement forward model, not a CFD solver.
    """
    rng=np.random.default_rng(seed)
    y,x=np.mgrid[0:size,0:size]
    sx=source_xy[0]*(size-1); sy=source_xy[1]*(size-1)
    dx=(x-sx)*pixel_size_m
    dy_north=-(y-sy)*pixel_size_m
    th=np.deg2rad(wind_dir_deg)
    wx=np.cos(th); wy=np.sin(th)
    down=dx*wx+dy_north*wy
    cross=-dx*wy+dy_north*wx

    sigma0=1.5*pixel_size_m
    sigma=sigma0 + dispersion*np.sqrt(np.maximum(down,0)+1.0)*5.0
    amp=(q_kg_h/max(wind_speed,0.5))*0.8
    plume=np.zeros_like(down,dtype=np.float64)
    mask=down>=0
    plume[mask]=amp/(np.maximum(sigma[mask],1.0)) * np.exp(-0.5*(cross[mask]/sigma[mask])**2)
    # Approximate longitudinal decay / dilution
    plume[mask]*=np.exp(-down[mask]/30000.0)
    plume=gaussian_filter(plume,0.8)

    background=gaussian_filter(rng.normal(0,1,(size,size)),5)
    background=background/(np.std(background)+1e-8)
    obs=np.maximum(plume + noise*max(np.max(plume),1e-6)*background,0)

    valid=np.ones((size,size),dtype=np.float32)
    if cloud_fraction>0:
        cloud=rng.random((size,size))<cloud_fraction
        cloud=gaussian_filter(cloud.astype(float),2)>0.3
        valid[cloud]=0
        obs[cloud]=0

    seg=(plume>max(np.max(plume)*0.05,1e-12)).astype(np.float32)*valid
    u=wind_speed*np.cos(th)
    v=wind_speed*np.sin(th)
    return {
        "enhancement":obs.astype(np.float32),
        "clean_plume":plume.astype(np.float32),
        "mask":seg.astype(np.float32),
        "valid":valid,
        "u10":float(u),"v10":float(v),
        "source_xy":np.asarray(source_xy,dtype=np.float32),
        "q_kg_h":float(q_kg_h),
    }
