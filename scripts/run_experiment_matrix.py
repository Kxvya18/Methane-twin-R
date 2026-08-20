"""
Research experiment orchestration.

This script creates isolated variant directories and invokes training/evaluation.
For large GPU experiments use a scheduler; this local runner is intentionally serial.
"""
import argparse,copy,json,subprocess,sys,tempfile
from pathlib import Path
import yaml

VARIANTS=[
    ("full",{}),
    ("no_physics",{"loss.physics":0.0}),
    ("low_physics",{"loss.physics":0.05}),
]
SEEDS=[7,42,137]

def set_nested(d,key,value):
    parts=key.split(".")
    for p in parts[:-1]: d=d[p]
    d[parts[-1]]=value

def main():
    ap=argparse.ArgumentParser();ap.add_argument("--config",default="configs/research.yaml");args=ap.parse_args()
    base=yaml.safe_load(Path(args.config).read_text())
    results=[]
    for seed in SEEDS:
        for name,updates in VARIANTS:
            cfg=copy.deepcopy(base);cfg["seed"]=seed
            cfg["paths"]["artifacts"]=f"artifacts/experiments/{name}/seed_{seed}"
            for k,v in updates.items():set_nested(cfg,k,v)
            with tempfile.NamedTemporaryFile("w",suffix=".yaml",delete=False) as f:
                yaml.safe_dump(cfg,f);cpath=f.name
            for cmd in (["train","pretrain"],["train","finetune"],["train","calibrate"],["evaluate","final"]):
                subprocess.run([sys.executable,"-m","methanetwin_r.cli",*cmd,"--config",cpath],check=True)
            metric=Path(cfg["paths"]["artifacts"])/"evaluation"/"final_test_metrics.json"
            results.append({"seed":seed,"variant":name,"metrics":json.loads(metric.read_text())})
    out=Path("artifacts/experiments/summary.json");out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(results,indent=2))
if __name__=="__main__":main()
