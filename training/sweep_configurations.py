import math

sweep_configurations = {
    "llm-bayes-invsqrtlr": {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "best_val_loss"},
        "parameters": {
            "lr_scheduler.name": {"value": "InverseSquareRootLR"},
            "optimizer.lr": {
                "distribution": "log_normal",
                "mu": math.log(1e-4),
                "sigma": (abs(math.log(1e-4) - math.log(3e-4))) ** 0.5,
            },
            "lr_scheduler.warm_up_factor": {"value": 1e-2},
            "accumulate_grad_batches": {
                "distribution": "q_normal",
                "mu": 16,
                "sigma": 4,
            },
        },
    },
    "bayes-constantlr": {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "best_val_loss"},
        "parameters": {
            "lr_scheduler.name": {"value": "ConstantLR"},
            "optimizer.lr": {
                "distribution": "log_normal",
                "mu": math.log(2e-4),
                "sigma": (abs(math.log(3e-4) - math.log(1e-4))) ** 0.5,
            },
            "active_layers.encoder": {
                "distribution": "q_normal",
                "mu": 6,
                "sigma": 2,
            },
            "active_layers.decoder": {
                "distribution": "q_normal",
                "mu": 6,
                "sigma": 2,
            },
        },
    },
    "bayes-invsqrtlr": {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "best_val_loss"},
        "parameters": {
            "lr_scheduler.name": {"value": "InverseSquareRootLR"},
            "optimizer.lr": {
                "distribution": "log_normal",
                "mu": math.log(2e-4),
                "sigma": (abs(math.log(3e-4) - math.log(1e-4))) ** 0.5,
            },
            "lr_scheduler.warm_up_factor": {
                "distribution": "normal",
                "mu": 2e-3,
                "sigma": 7e-4**0.5,
            },
            "active_layers.encoder": {
                "distribution": "q_normal",
                "mu": 6,
                "sigma": 2,
            },
            "active_layers.decoder": {
                "distribution": "q_normal",
                "mu": 6,
                "sigma": 2,
            },
        },
    },
    "bayes-cycliclr": {
        "method": "bayes",
        "metric": {"goal": "minimize", "name": "best_val_loss"},
        "parameters": {
            "lr_scheduler.name": {"value": "CyclicLR"},
            "optimizer.lr": {
                "distribution": "log_normal",
                "mu": math.log(1e-4),
                "sigma": (abs(math.log(1e-4) - math.log(2e-4))) ** 0.5,
            },
            "lr_scheduler.step_size_up": {
                "distribution": "q_normal",
                "mu": 200,
                "sigma": 40**0.5,
            },
            "active_layers.encoder": {
                "distribution": "q_normal",
                "mu": 6,
                "sigma": 2,
            },
            "active_layers.decoder": {
                "distribution": "q_normal",
                "mu": 6,
                "sigma": 2,
            },
        },
    },
}
