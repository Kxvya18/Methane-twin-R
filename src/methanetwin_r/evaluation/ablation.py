from pathlib import Path
from ..utils import write_json

def run_ablation_plan(cfg):
    """
    Creates a reproducible ablation plan. Full retraining is intentionally explicit:
    every variant gets its own checkpoint/results directory rather than silently
    mutating a single model.
    """
    variants=[
        {"name":"full","wind":True,"physics":True,"attention":"wind","pretraining":True},
        {"name":"no_wind","wind":False,"physics":True,"attention":"standard","pretraining":True},
        {"name":"no_physics","wind":True,"physics":False,"attention":"wind","pretraining":True},
        {"name":"standard_attention","wind":True,"physics":True,"attention":"standard","pretraining":True},
        {"name":"no_pretraining","wind":True,"physics":True,"attention":"wind","pretraining":False},
    ]
    out=Path(cfg["paths"]["artifacts"])/"ablation_plan.json";write_json(out,variants)
    return {"variants":variants,"note":"Use scripts/run_experiment_matrix.py to execute retraining variants."}
