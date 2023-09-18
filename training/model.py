import torch
from lightning import pytorch as pl
from typing import Optional
import numpy as np
import random
import warnings

import training.utils as utils
from helpers.review_set import ReviewSet
from helpers.label_selection import (
    DatasetSelectionStrategy,
    LabelSelectionStrategyInterface,
)
from transformers.modeling_outputs import Seq2SeqLMOutput


NUM_WORKERS = 1


class ReviewModel(pl.LightningModule):
    def __init__(
        self,
        model,
        active_layers: str,
        model_name: str,
        tokenizer,
        max_length: int,
        optimizer,
        hyperparameters: dict,
        dataset_config: dict,
        trainer: pl.Trainer,
        multiple_usage_options_strategy: str,
        optimizer_args: dict,
        lr_scheduler_args: dict,
        gradual_unfreezing_mode: Optional[str],
        prompt_id: str,
    ):
        super(ReviewModel, self).__init__()
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.optimizer = optimizer
        self.dataset_config = dataset_config
        self.hyperparameters = hyperparameters
        self.active_layers = active_layers
        self.trainer = trainer
        self.multiple_usage_options_strategy = multiple_usage_options_strategy
        self.optimizer_args = optimizer_args
        self.lr_scheduler_args = lr_scheduler_args
        self.prompt_id = prompt_id
        self.gradual_unfreezing_mode = gradual_unfreezing_mode
        self.validation_loss = []
        self.tokenization_args = {
            "tokenizer": tokenizer,
            "model_max_length": max_length,
            "for_training": True,
        }

        self._initialize_datasets()

        # Skip freezing if a fake test model is loaded
        if self.model is not None:
            self.active_encoder_layers, self.active_decoder_layers = utils.freeze_model(
                active_layers, model
            )

    def run_name(self) -> str:
        if self.trainer is not None and self.trainer.logger is not None:
            return self.trainer.logger.experiment.name
        return "test-run"

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
    ):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )

    def _step(self, batch) -> Seq2SeqLMOutput:
        # self() calls self.forward(), but should be preferred (https://github.com/Lightning-AI/lightning/issues/1209)
        labels = batch["output"]["input_ids"]
        outputs = self(
            input_ids=batch["input"]["input_ids"],
            attention_mask=batch["input"]["attention_mask"],
            labels=labels,
        )
        return outputs

    def training_step(self, batch, batch_idx):
        outputs = self._step(batch)
        self.log(
            "train_loss",
            outputs.loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.hyperparameters["batch_size"],
        )
        if hasattr(self, "lr_scheduler"):
            self.log(
                "lr", self.lr_scheduler.get_last_lr()[0], on_step=True, logger=True
            )
        return outputs.loss

    def training_epoch_end(self, outputs):
        """Logs the average training loss over the epoch"""
        avg_loss = torch.stack(
            [x["loss"] * self.trainer.accumulate_grad_batches for x in outputs]
        ).mean()
        self.log(
            "epoch_train_loss",
            avg_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.hyperparameters["batch_size"],
        )

        self.log(
            "epoch_end_lr",
            self.hyperparameters["lr"]
            if not hasattr(self, "lr_scheduler")
            else self.lr_scheduler.get_last_lr()[0],
            on_epoch=True,
            logger=True,
            sync_dist=True,
            batch_size=self.hyperparameters["batch_size"],
        )

        utils.gradual_unfreeze(
            self.model,
            self.current_epoch,
            self.gradual_unfreezing_mode,
            self.active_encoder_layers,
            self.active_decoder_layers,
        )

    def validation_step(self, batch, batch_idx):
        outputs = self._step(batch)
        self.log(
            "val_loss",
            outputs.loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.hyperparameters["batch_size"],
        )
        return outputs.loss

    def validation_epoch_end(self, outputs):
        """Logs the average validation loss over the epoch"""
        avg_loss = torch.stack(outputs).mean()
        self.log(
            "epoch_val_loss",
            avg_loss,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
            batch_size=self.hyperparameters["batch_size"],
        )
        self.validation_loss.append(avg_loss.item())

    def test_step(self, batch, __):
        outputs = self._step(batch)
        self.log(
            "test_loss",
            outputs.loss,
            on_step=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        return outputs.loss

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), **self.optimizer_args)
        scheduler_name = self.lr_scheduler_args["name"]

        if scheduler_name == "OneCycleLR":
            self.lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.hyperparameters["lr"],
                total_steps=self.trainer.estimated_stepping_batches,
            )
            return [optimizer], [{"scheduler": self.lr_scheduler, "interval": "step"}]

        if scheduler_name == "AdaFactor":
            from transformers.optimization import AdafactorSchedule

            self.lr_scheduler = AdafactorSchedule(optimizer)
            return [optimizer], [{"scheduler": self.lr_scheduler, "interval": "step"}]

        if scheduler_name == "CyclicLR":
            self.lr_scheduler = torch.optim.lr_scheduler.CyclicLR(
                optimizer,
                mode=self.lr_scheduler_args["mode"],
                base_lr=self.hyperparameters["lr"],
                max_lr=self.hyperparameters["lr"] * 5,
                step_size_up=max(0, self.lr_scheduler_args["step_size_up"]),
                cycle_momentum="momentum" in self.optimizer_args,
            )
            return [optimizer], [{"scheduler": self.lr_scheduler, "interval": "step"}]

        if scheduler_name == "InverseSquareRootLR":
            warm_up_factor = max(1e-10, self.lr_scheduler_args["warm_up_factor"])
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda step: 1e1
                * min(
                    warm_up_factor * (step + 1),
                    1 / ((step + 1) ** 0.5),
                ),
            )
            return [optimizer], [{"scheduler": self.lr_scheduler, "interval": "step"}]

        if scheduler_name == "ConstantLr":
            return [optimizer]

        if scheduler_name == "LRRangeTest":
            self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lambda epoch: 1e-1 * ((100**0.25) ** (epoch - 1)),
            )
            return [optimizer], [{"scheduler": self.lr_scheduler, "interval": "epoch"}]

        # This is the right way to return an optimizer, without scheduler, in lightning (https://github.com/Lightning-AI/lightning/issues/3795)
        return [optimizer]

    def get_best_epoch(self) -> Optional[int]:
        try:
            return int(
                np.argmin(self.validation_loss[1:])
            )  # The first element is the validation loss during dataloader sanity check, which is not relevant.
        except ValueError:
            return None

    def _print_dataloader_stats(self, metadata: dict) -> None:
        print("".join([f"\t{k}: {v}\n" for k, v in metadata.items()]))

    def general_dataloader_args(self):
        return {
            "batch_size": self.hyperparameters["batch_size"],
            "num_workers": NUM_WORKERS,
            "multiple_usage_options_strategy": self.multiple_usage_options_strategy,
            "prompt_id": self.prompt_id,
        }

    def train_dataloader(self):
        training_set_config = self.dataset_config["training_set"]
        dataloader, metadata = self.train_reviews.get_dataloader(
            selection_strategy=self.train_reviews_strategy,
            drop_out=training_set_config["drop_out"],
            stratified_drop_out=training_set_config["stratified_drop_out"],
            rng=random.Random(training_set_config["dataloader_setup_seed"]),
            **self.tokenization_args,
            drop_last=True,  # Drops the last incomplete batch, if the dataset size is not divisible by the batch size.
            shuffle=True,  # Shuffles the training data every epoch.
            pin_memory=True,
            **self.general_dataloader_args(),
        )
        print("\n\nTrain dataloader stats:")
        self._print_dataloader_stats(metadata | {"num_batches": len(dataloader)})
        return dataloader

    def val_dataloader(self):
        dataloader, metadata = self.val_reviews.get_dataloader(
            selection_strategy=self.val_reviews_strategy,
            **self.tokenization_args,
            **self.general_dataloader_args(),
        )
        print("\n\nValidation dataloader stats:")
        self._print_dataloader_stats(metadata | {"num_batches": len(dataloader)})
        return dataloader

    def test_dataloader(self):
        if not self.test_reviews:
            return torch.utils.data.DataLoader([])

        dataloader, metadata = self.test_reviews.get_dataloader(
            selection_strategy=self.test_reviews_strategy,
            **self.tokenization_args,
            **self.general_dataloader_args(),
        )
        print("\n\nTest dataloader stats:")
        self._print_dataloader_stats(metadata | {"num_batches": len(dataloader)})
        return dataloader

    def _prepare_training_set(
        self, training_set_config: dict, test_reviews: ReviewSet
    ) -> tuple[ReviewSet, LabelSelectionStrategyInterface]:
        train_reviews_strategy = DatasetSelectionStrategy(training_set_config["name"])
        reviews = (
            ReviewSet.from_files(utils.get_dataset_path(training_set_config["name"]))
            - test_reviews
        )
        return reviews, train_reviews_strategy

    def _initialize_datasets(self):
        test_set_name = self.dataset_config["test_set"]["name"]
        if test_set_name:
            self.test_reviews_strategy = DatasetSelectionStrategy(test_set_name)
            self.test_reviews = ReviewSet.from_files(
                utils.get_dataset_path(test_set_name)
            )
        else:
            self.test_reviews_strategy = None
            self.test_reviews = ReviewSet.from_reviews()

        training_set_config = self.dataset_config["training_set"]
        train_reviews, self.train_reviews_strategy = self._prepare_training_set(
            training_set_config, self.test_reviews
        )

        validation_set_name = self.dataset_config["validation_set"]["name"]
        if validation_set_name:
            self.val_reviews_strategy = DatasetSelectionStrategy(validation_set_name)
            self.val_reviews = (
                ReviewSet.from_files(utils.get_dataset_path(validation_set_name))
                - self.test_reviews
            )
            self.train_reviews = train_reviews - self.val_reviews
        else:
            self.val_reviews, self.train_reviews = train_reviews.stratified_split(
                training_set_config["validation_split"],
                self.train_reviews_strategy,
            )
            self.val_reviews_strategy = self.train_reviews_strategy

        if (
            len(self.train_reviews) == 0
            or len(self.val_reviews) == 0
            or len(self.test_reviews) == 0
        ):
            warnings.warn(
                "One of the reviewsets (train, val or test) is empty. Please check your dataset configuration."
            )

    def on_save_checkpoint(self, checkpoint):
        checkpoint["model_name"] = self.model_name
        return checkpoint
