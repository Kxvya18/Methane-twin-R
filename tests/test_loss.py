import pytest
torch=pytest.importorskip("torch")
from methanetwin_r.models.network import build_model
from methanetwin_r.models.loss import multitask_loss

def test_loss_finite():
    m=build_model(base_channels=8,attention_heads=4,dropout=0.1)
    x=torch.rand(2,4,32,32)
    o=m(x)
    b={"x":x,"mask":torch.zeros(2,1,32,32),"detection":torch.tensor([1.,0.]),
       "source":torch.tensor([[.2,.5],[float("nan"),float("nan")]]),
       "source_weight":torch.tensor([1.,0.]),"q":torch.tensor([1000.,float("nan")]),
       "q_available":torch.tensor([1.,0.])}
    w={"detection":1,"segmentation":1,"source":1,"emission":1,"physics":0}
    loss,_=multitask_loss(o,b,w)
    assert torch.isfinite(loss)
