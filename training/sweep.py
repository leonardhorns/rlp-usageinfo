#!/usr/bin/env python3
import sys
import wandb

from training.train import train
from training.sweep_configurations import sweep_configurations

if len(sys.argv) < 2:
    print("Usage: python sweep.py sweep_name")
    sys.exit(1)

sweep_name = sys.argv.pop(1)
sweep_configuration = sweep_configurations[sweep_name] | {"name": sweep_name}

sweep_id = wandb.sweep(sweep=sweep_configuration)

count = None
for arg in sys.argv[1:]:
    if arg.startswith("--count="):
        count = int(arg.split("=")[1])
        sys.argv.remove(arg)
    elif arg.startswith("--tag="):
        tag = arg.split("=")[1]
        sys.argv.remove(arg)

wandb.agent(
    sweep_id,
    function=lambda: train(
        is_sweep=True, run_name=f"{sweep_configuration['name']}-{tag}"
    ),
    count=count,
)
