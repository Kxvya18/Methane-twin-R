import argparse,subprocess,sys
from pathlib import Path

STAGES=[
    ["data","download-emit"],
    ["data","catalog"],
    ["data","metadata-report"],
    ["data","download-era5"],
    ["data","build"],
    ["physics","generate"],
    ["train","baselines"],
    ["train","tune"],
    ["train","pretrain"],
    ["train","finetune"],
    ["train","calibrate"],
    ["evaluate","final"],
]

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--config",default="configs/research.yaml")
    p.add_argument("--skip-downloads",action="store_true")
    args=p.parse_args()
    stages=STAGES[2:] if args.skip_downloads else STAGES
    for stage in stages:
        cmd=[sys.executable,"-m","methanetwin_r.cli",*stage,"--config",args.config]
        print("\n>>>"," ".join(cmd),flush=True)
        subprocess.run(cmd,check=True)

if __name__=="__main__":main()
