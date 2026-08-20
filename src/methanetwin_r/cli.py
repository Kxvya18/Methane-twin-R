import argparse,json
from .config import load_config

def main():
    p=argparse.ArgumentParser(prog="methanetwin")
    sub=p.add_subparsers(dest="group",required=True)

    data=sub.add_parser("data"); dsub=data.add_subparsers(dest="command",required=True)
    for name in ["download-emit","catalog","metadata-report","download-era5","build"]:
        q=dsub.add_parser(name); q.add_argument("--config",default="configs/research.yaml")

    phys=sub.add_parser("physics"); psub=phys.add_subparsers(dest="command",required=True)
    q=psub.add_parser("generate"); q.add_argument("--config",default="configs/research.yaml")

    train=sub.add_parser("train"); tsub=train.add_subparsers(dest="command",required=True)
    for name in ["baselines","tune","pretrain","finetune","calibrate"]:
        q=tsub.add_parser(name);q.add_argument("--config",default="configs/research.yaml")

    attr=sub.add_parser("attribute"); attr.add_argument("--config",default="configs/research.yaml")
    pers=sub.add_parser("persistence"); pers.add_argument("--config",default="configs/research.yaml")

    ev=sub.add_parser("evaluate"); ev.add_argument("--config",default="configs/research.yaml")
    ev.add_argument("command",nargs="?",default="final",choices=["final","ablate"])

    args=p.parse_args();cfg=load_config(args.config)

    if args.group=="data":
        if args.command=="download-emit":
            from .data.emit import download_emit_from_config; out=download_emit_from_config(cfg)
        elif args.command=="catalog":
            from .data.catalog import build_emit_catalog;out=build_emit_catalog(cfg)
        elif args.command=="metadata-report":
            from .data.catalog import metadata_report;out=metadata_report(cfg)
        elif args.command=="download-era5":
            from .data.era5 import download_era5_for_events;out=download_era5_for_events(cfg)
        elif args.command=="build":
            from .data.build import build_real_dataset;out=build_real_dataset(cfg)
    elif args.group=="physics":
        from .simulation.generate import generate_physics_dataset;out=generate_physics_dataset(cfg)
    elif args.group=="train":
        if args.command=="baselines":
            from .models.baselines import train_baselines;out=train_baselines(cfg)
        elif args.command=="tune":
            from .models.pipeline import tune;out=tune(cfg)
        elif args.command=="pretrain":
            from .models.pipeline import pretrain;out=pretrain(cfg)
        elif args.command=="finetune":
            from .models.pipeline import finetune;out=finetune(cfg)
        elif args.command=="calibrate":
            from .evaluation.calibration import calibrate;out=calibrate(cfg)
    elif args.group=="attribute":
        from .attribution.run import run_attribution;out=run_attribution(cfg)
    elif args.group=="persistence":
        from .attribution.persistence_run import run_persistence;out=run_persistence(cfg)
    elif args.group=="evaluate":
        if args.command=="ablate":
            from .evaluation.ablation import run_ablation_plan;out=run_ablation_plan(cfg)
        else:
            from .evaluation.final import evaluate_final;out=evaluate_final(cfg)
    print(json.dumps(out,indent=2,default=str))

if __name__=="__main__":main()
