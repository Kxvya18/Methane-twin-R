import numpy as np
from methanetwin_r.features.plume import plume_features,FEATURE_NAMES

def test_features_finite():
    a=np.zeros((32,32));a[14:18,10:25]=1
    f=plume_features(a,a>0,u=5,v=0)
    assert len(f)==len(FEATURE_NAMES)
    assert np.isfinite(f).all()
