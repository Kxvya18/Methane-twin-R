import numpy as np

def plume_features(img, mask=None, u=0.0, v=0.0):
    x=np.nan_to_num(np.asarray(img,float),nan=0.0)
    pos=np.maximum(x,0)
    if mask is None:
        thr=np.quantile(pos[pos>0],0.75) if (pos>0).any() else np.inf
        mask=pos>=thr
    mask=np.asarray(mask)>0
    ys,xs=np.where(mask)
    w=pos[mask]
    if len(xs)==0 or w.sum()<=0:
        cx=cy=0.5; elong=0; orient=0; area=0; down_extent=0
    else:
        cx=float(np.average(xs,weights=w)/(x.shape[1]-1))
        cy=float(np.average(ys,weights=w)/(x.shape[0]-1))
        pts=np.c_[xs-np.average(xs,weights=w), -(ys-np.average(ys,weights=w))]
        cov=(pts*w[:,None]).T@pts/max(w.sum(),1e-9)
        eigval,eigvec=np.linalg.eigh(cov)
        order=np.argsort(eigval)[::-1]
        eigval=eigval[order]; vec=eigvec[:,order[0]]
        elong=float(np.sqrt(max(eigval[0],0)/max(eigval[-1],1e-8)))
        orient=float(np.arctan2(vec[1],vec[0]))
        area=float(mask.mean())
        speed=max(np.hypot(u,v),1e-8)
        wind=np.array([u/speed,v/speed])
        projections=pts@wind
        down_extent=float(np.ptp(projections)) if len(projections) else 0

    gy,gx=np.gradient(pos)
    speed=max(np.hypot(u,v),1e-8)
    wind_angle=float(np.arctan2(v,u))
    alignment=float(np.cos(orient-wind_angle))
    return np.array([
        float(np.max(pos)),float(np.mean(pos)),float(np.std(pos)),float(np.sum(pos)),
        float(np.quantile(pos,0.95)),area,cx,cy,elong,np.sin(orient),np.cos(orient),
        float(np.mean(np.hypot(gx,gy))),float(u),float(v),float(speed),alignment,down_extent
    ],dtype=np.float32)

FEATURE_NAMES=["max","mean","std","sum","p95","area","cx","cy","elongation",
               "sin_orientation","cos_orientation","mean_gradient","u10","v10","wind_speed",
               "wind_alignment","downwind_extent"]
