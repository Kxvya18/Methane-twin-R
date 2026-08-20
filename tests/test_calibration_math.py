import numpy as np
from methanetwin_r.evaluation.metrics import haversine_km

def test_haversine_zero():
    assert haversine_km(10,20,10,20)<1e-8
