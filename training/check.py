"""
Run this on your Jupyter server:
    python3 inspect_configs.py
"""

import yaml
import torch
from pathlib import Path
from omegaconf import OmegaConf, DictConfig

CHECKPOINT_DIR = Path("/mnt/weatherloss/WindPower/training/EGU26/checkpoint")

RUNS = {
    "NoPower2":        CHECKPOINT_DIR / "NoPower2",
    "VanillaPower":    CHECKPOINT_DIR / "VanillaPower",
    "LowPower":        CHECKPOINT_DIR / "LowPower",
    "Finetune":        CHECKPOINT_DIR / "Finetune",
    "PowerZeroWeight": CHECKPOINT_DIR / "PowerZeroWeight",
}

def find_training_ckpt(run_dir: Path) -> Path | None:
    """Prefer smallest training ckpt to load fast (config is same in all)."""
    candidates = sorted(run_dir.glob("anemoi-by_epoch-epoch_000*.ckpt"))
    if not candidates:
        candidates = sorted(run_dir.glob("anemoi-by_epoch*.ckpt"))
    return candidates[0] if candidates else None

def load_config(ckpt_path: Path) -> DictConfig | None:
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    # Training ckpts are plain dicts with hyper_parameters key
    if not isinstance(raw, dict):
        print(f"  Unexpected type: {type(raw)}")
        return None

    hp = raw.get("hyper_parameters", {})
    print(f"  hyper_parameters keys: {list(hp.keys())}")

    # Config is usually directly in hyper_parameters or under 'config'
    cfg = hp.get("config", hp)
    if cfg is None:
        print("  No config found in hyper_parameters")
        return None

    if isinstance(cfg, DictConfig):
        return cfg
    try:
        return OmegaConf.create(cfg)
    except Exception as e:
        print(f"  OmegaConf.create failed: {e}")
        return None

def safe_get(cfg, *keys):
    try:
        node = cfg
        for k in keys:
            node = node[k]
        try:
            return OmegaConf.to_container(node, resolve=False)
        except Exception:
            return node
    except Exception:
        return "<not found>"

FIELDS = [
    ("training", "variable_loss_scaling"),
    ("training", "scalers"),
    ("training", "training_loss"),
    ("training", "fork_run_id"),
    ("training", "run_id"),
    ("training", "load_weights_only"),
    ("training", "multistep_input"),
    ("training", "rollout"),
    ("data", "forcings"),
    ("data", "diagnostic"),
]

def main():
    run_data = {}

    for run_name, run_dir in RUNS.items():
        print(f"\n{'='*60}")
        print(f"Run: {run_name}")
        if not run_dir.exists():
            print("  Dir not found"); continue

        ckpt_path = find_training_ckpt(run_dir)
        if not ckpt_path:
            print("  No training ckpt found"); continue

        print(f"  File: {ckpt_path.name}")
        cfg = load_config(ckpt_path)
        if cfg is None:
            print("  Config extraction failed"); continue

        print("  OK")
        run_data[run_name] = cfg

    if not run_data:
        print("\nFailed to load any configs.")
        return

    print("\n\n" + "#"*70)
    print("# COMPARISON (*** = differs between runs)")
    print("#"*70)

    for field_path in FIELDS:
        label = " > ".join(field_path)
        values = {n: safe_get(cfg, *field_path) for n, cfg in run_data.items()}
        differs = len(set(str(v) for v in values.values())) > 1

        print(f"\n{'─'*60}")
        print(f"  {label}{'  *** DIFFERS ***' if differs else ''}")
        print(f"{'─'*60}")
        for rname, val in values.items():
            print(f"\n  [{rname}]")
            if isinstance(val, (dict, list)):
                for line in yaml.dump(val, default_flow_style=False).splitlines():
                    print(f"    {line}")
            else:
                print(f"    {val}")

    # Full configs for ZeroWeight vs baseline
    for run_name in ["PowerZeroWeight", "NoPower2"]:
        cfg = run_data.get(run_name)
        if cfg:
            print(f"\n\n{'#'*70}")
            print(f"# FULL CONFIG — {run_name}")
            print(f"{'#'*70}")
            try:
                print(OmegaConf.to_yaml(cfg))
            except Exception as e:
                print(f"(OmegaConf.to_yaml failed: {e})")
                print(yaml.dump(OmegaConf.to_container(cfg, resolve=False), default_flow_style=False))

if __name__ == "__main__":
    main()