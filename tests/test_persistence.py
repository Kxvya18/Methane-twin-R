from methanetwin_r.attribution.persistence import beta_persistence

def test_more_detections_means_higher_persistence():
    a=beta_persistence(1,10)
    b=beta_persistence(8,10)
    assert b["posterior_mean"]>a["posterior_mean"]
