#!/usr/bin/env python3
# %%
import sys, os
import warnings
from copy import copy
import torch
import wandb
from lightning import pytorch as pl
from pprint import pprint
from datetime import datetime

from model import ReviewModel
from helpers.sustainability_logger import SustainabilityLogger
from generator import DEFAULT_GENERATION_CONFIG, Generator
import utils
from helpers.review_set import ReviewSet


def get_cli_args():
    # Command line args
    args_config = {}
    for arg in sys.argv[1:]:
        if not arg.startswith("--") or "=" not in arg:
            print(f"Unrecognized argument: {arg}")
            print(
                "Please only provide arguments in the form --key=value or --key.key.key...=value (for nested parameters)"
            )
            exit(1)

        key, value = arg[2:].split("=")
        args_config[key] = value

    return args_config


def convert_active_layer_parameters(config):
    for key, value in copy(config).items():
        if (
            key.startswith("active_layers.")
            and key != "active_layers.lm_head"
            and type(value) == int
        ):
            config[key] = ":0" if value <= 0 else f"-{value}:"


def convert_to_correct_type(value, prev_type):
    if value == "" or value is None:
        return None
    return (prev_type or str)(value)


def update_config_values(base_config, update_values, delimiter="."):
    for key, value in update_values.copy().items():
        try:
            if delimiter in key:
                del update_values[key]
                keys = key.split(delimiter)
                current_key = keys.pop(0)
                update_values[current_key] = update_values.get(
                    current_key, copy(base_config[current_key])
                )
                current_config = update_values[current_key]
                while len(keys) > 1:
                    current_key = keys.pop(0)
                    current_config = current_config[current_key]
                value_type = type(current_config[keys[0]])
                current_config[keys[0]] = convert_to_correct_type(value, value_type)
            else:
                update_values[key] = convert_to_correct_type(
                    value, type(base_config[key])
                )
        except (KeyError, ValueError) as e:
            if isinstance(e, KeyError):
                print(f"Unknown config key: {key}\n{e}")
                exit(1)
            else:
                print(f"Invalid value for {key}: {value}")
                print(f"Expected type: {type(base_config[key])}")
                exit(1)

    return base_config | update_values


def train(is_sweep=False, run_name=None):
    torch.set_float32_matmul_precision("medium")
    warnings.filterwarnings(
        "ignore", ".*Consider increasing the value of the `num_workers` argument*"
    )

    # Initialize config
    cli_args_config = get_cli_args()
    config = utils.get_config(
        cli_args_config.pop("config", None) or utils.get_config_path("training_config")
    )
    config = update_config_values(config, cli_args_config)

    if is_sweep or not config["test_run"]:
        logger = pl.loggers.WandbLogger(
            project="rlp-t2t",
            entity="bsc2022-usageinfo",
            name=f"{run_name}-{datetime.now().strftime('%m%d%H%M%S')}"
            if run_name
            else None,
        )
        wandb_config = dict(wandb.config)
        convert_active_layer_parameters(wandb_config)
        config = update_config_values(config, wandb_config, delimiter=".")

        checkpoint_callback = utils.get_checkpoint_callback(logger, config)
    early_stopping_callback = pl.callbacks.early_stopping.EarlyStopping(
        monitor="epoch_val_loss", patience=5
    )

    print("----------------------------------\nTraining config:")
    pprint(config)
    print("----------------------------------")

    # the method below will also check if either model_name or artifact is provided
    model, tokenizer, max_length, model_name = utils.initialize_model_tuple(
        config["model_name"]
    )

    test_run = config["test_run"]
    files_to_generate_on = list(
        filter(
            lambda file: file is not None and os.path.exists(str(file)),
            config["files_to_generate_on"],
        )
    )
    del config["test_run"], config["files_to_generate_on"]

    hyperparameters = {
        "weight_decay": config["optimizer"]["weight_decay"],
        "batch_size": config["batch_size"],
        "lr": config["optimizer"]["lr"],
    }
    dataset_parameters = copy(config["dataset"])
    optimizer, optimizer_args = utils.get_optimizer(config["optimizer"])
    for key in copy(config["optimizer"]):
        if key not in optimizer_args and key != "name":
            del config["optimizer"][key]

    if is_sweep or not test_run:
        logger.experiment.config.update(config)

    pl.seed_everything(seed=config["seed"], workers=True)

    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available, using CPU instead.")

    trainer = pl.Trainer(
        deterministic=True,
        max_epochs=config["epochs"] or None,
        accelerator="auto",
        callbacks=[checkpoint_callback, early_stopping_callback]
        if not test_run
        else [early_stopping_callback],
        logger=logger if not test_run else None,
        accumulate_grad_batches=max(1, config["accumulate_grad_batches"]),
    )

    model = ReviewModel(
        model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        max_length=max_length,
        active_layers=config["active_layers"],
        optimizer=optimizer,
        optimizer_args=optimizer_args,
        lr_scheduler_args=config["lr_scheduler"],
        hyperparameters=hyperparameters,
        dataset_config=dataset_parameters,
        trainer=trainer,
        multiple_usage_options_strategy=config["multiple_usage_options_strategy"],
        gradual_unfreezing_mode=config["gradual_unfreezing_mode"],
        prompt_id=config["prompt_id"],
    )

    # %% Training and testing
    if not test_run:
        with SustainabilityLogger(description="training"):
            trainer.fit(model)
        with SustainabilityLogger(description="testing"):
            trainer.test()

        wandb.log({"best_val_loss": checkpoint_callback.best_model_score})

        try:
            label_id = f"model-{wandb.run.name}-auto"

            generator = Generator(
                wandb.run.name, DEFAULT_GENERATION_CONFIG, checkpoint="best"
            )
            for file in files_to_generate_on:
                print(file)
                review_set = ReviewSet.from_files(file)
                generator.generate_label(review_set, label_id=label_id, verbose=True)

                review_set.save()
        except Exception as e:
            warnings.warn(
                "Could not generate label for the dataset. The run has probably failed.",
                e,
            )
        finally:
            wandb.finish()
    else:
        trainer.fit(model)
        trainer.test()


if __name__ == "__main__":
    wandb.login()

    train()
