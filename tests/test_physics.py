import numpy as np
from methanetwin_r.physics.plume import gaussian_plume_2d
from methanetwin_r.physics.inversion import estimate_q_given_source

def test_plume_is_downwind_and_positive():
    d=gaussian_plume_2d(size=64,source_xy=(0.25,0.5),q_kg_h=1000,wind_speed=5,wind_dir_deg=0,noise=0)
    assert d["enhancement"].max()>0
    assert d["mask"].sum()>0
    left=d["clean_plume"][:,:16].sum()
    right=d["clean_plume"][:,16:].sum()
    assert right>left

def test_q_inverse_recovers_scale():
    d=gaussian_plume_2d(size=64,source_xy=(0.3,0.5),q_kg_h=1200,wind_speed=4,wind_dir_deg=20,noise=0)
    q=estimate_q_given_source(d["clean_plume"],(0.3,0.5),4,20)
    assert abs(q-1200)/1200<0.05
