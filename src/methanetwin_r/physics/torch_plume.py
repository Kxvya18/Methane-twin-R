def differentiable_plume(batch_size, size, source_xy, q, u, v, device):
    import torch
    ys=torch.linspace(0,1,size,device=device)
    xs=torch.linspace(0,1,size,device=device)
    yy,xx=torch.meshgrid(ys,xs,indexing="ij")
    xx=xx[None].expand(batch_size,-1,-1)
    yy=yy[None].expand(batch_size,-1,-1)
    sx=source_xy[:,0,None,None]; sy=source_xy[:,1,None,None]
    speed=torch.sqrt(u*u+v*v).clamp_min(0.5)
    wx=u/speed; wy=v/speed
    dx=(xx-sx)*size
    dy=-(yy-sy)*size
    down=dx*wx[:,None,None]+dy*wy[:,None,None]
    cross=-dx*wy[:,None,None]+dy*wx[:,None,None]
    sigma=1.5+0.8*torch.sqrt(torch.relu(down)+1)
    amp=torch.exp(q)[:,None,None]/speed[:,None,None]
    plume=amp/sigma*torch.exp(-0.5*(cross/sigma)**2)
    plume=plume*torch.sigmoid(8*down)
    plume=plume/(plume.amax(dim=(1,2),keepdim=True).clamp_min(1e-6))
    return plume
