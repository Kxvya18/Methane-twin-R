import pytest

torch=pytest.importorskip("torch")
from methanetwin_r.models.network import build_model

def test_model_output_shapes():
    m=build_model(base_channels=8,attention_heads=4,dropout=0.1)
    x=torch.randn(2,4,64,64)
    o=m(x)
    assert o["mask_logit"].shape==(2,1,64,64)
    assert o["source_xy"].shape==(2,2)
    assert o["log_q"].shape==(2,)
