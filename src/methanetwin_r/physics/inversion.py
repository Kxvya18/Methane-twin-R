import numpy as np
from .plume import gaussian_plume_2d

def estimate_q_given_source(observed, source_xy, wind_speed, wind_dir_deg, pixel_size_m=60.0):
    template=gaussian_plume_2d(
        size=observed.shape[0],pixel_size_m=pixel_size_m,source_xy=source_xy,
        q_kg_h=1.0,wind_speed=wind_speed,wind_dir_deg=wind_dir_deg,noise=0
    )["clean_plume"]
    denom=float(np.sum(template*template))
    if denom<=1e-12:
        return 0.0
    return float(max(0,np.sum(observed*template)/denom))

def grid_search_inverse(observed, wind_speed, wind_dir_deg, grid=9, pixel_size_m=60.0):
    best=None
    for sy in np.linspace(0.1,0.9,grid):
        for sx in np.linspace(0.1,0.9,grid):
            q=estimate_q_given_source(observed,(sx,sy),wind_speed,wind_dir_deg,pixel_size_m)
            recon=gaussian_plume_2d(
                size=observed.shape[0],pixel_size_m=pixel_size_m,source_xy=(sx,sy),
                q_kg_h=q,wind_speed=wind_speed,wind_dir_deg=wind_dir_deg,noise=0
            )["clean_plume"]
            err=float(np.mean((observed-recon)**2))
            if best is None or err<best["mse"]:
                best={"source_xy":[float(sx),float(sy)],"q_kg_h":q,"mse":err}
    return best
