from collections import defaultdict
from torch.optim import SGD
from torch.optim.lr_scheduler import LambdaLR
import numpy
import pytorch_lightning
import torch

from src.modules.models_and_frameworks.fixmatch.fixmatch_components import (
    FixMatchDataAugmenter,
    FixMatchLossFunction,
    FixMatchModel,
    FixMatchPerformanceMetrics
)


class PyTorchLightningFixMatchFramework(pytorch_lightning.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.accumulated_batches = None
        self.config = None
        self.criterion = None
        self.data_augmenter = None
        self.model = None
        self.performance_metrics = None

        self._setup(**kwargs)

    def configure_optimizers(self):
        optimizer = SGD(self.parameters(), **self.config.optimizer.sgd)
        total_number_of_training_steps = self.config.optimizer.lr_scheduler \
            .number_of_training_steps_per_epoch \
            * self.config.optimizer.lr_scheduler.max_epochs
        scheduler = LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda current_training_step: numpy.cos(
                7 * numpy.pi * current_training_step
                / (16 * total_number_of_training_steps)
            )
        )
        optimizer_and_lr_scheduler_configuration = {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }
        return optimizer_and_lr_scheduler_configuration

    def forward(self, image):
        batch_of_logits = self.model(image)
        return batch_of_logits

    def training_step(self, batch, batch_idx):
        print(f"\ntraining_step, {batch_idx =}")
        print(f"{self.config.dataset_name.labeled = }")
        print(f"{self.config.dataset_name.unlabeled = }")
        labeled_batch = batch[self.config.dataset_name.labeled]
        unlabeled_batch = batch[self.config.dataset_name.unlabeled]

        if labeled_batch is not None:
            labeled_images = labeled_batch['data']['images']
            labels = labeled_batch['labels']
            labeled_batch_size = labels.shape[0]
            print(f"labeled_batch is not None, {labeled_batch_size = }")
        else:
            print("labeled_batch is None, labeled_batch_size = 0")
            labeled_images, labels = None, None
            labeled_batch_size = 0

        if unlabeled_batch is not None:
            unlabeled_images = unlabeled_batch['data']['images']
            unlabeled_batch_size = unlabeled_images.shape[0]
            print(f"unlabeled_batch is not None, {unlabeled_batch_size = }")
        else:
            print("unlabeled_batch is None, unlabeled_batch_size = 0")
            unlabeled_images = None
            unlabeled_batch_size = 0

        logits = self._get_logits_by_image_and_augmentation_type(
            labeled_images,
            unlabeled_images,
            training_step=True
        )
        loss, unlabeled_data_utilization = self.criterion(logits, labels)
        effective_batch_size = (
            labeled_batch_size
            + unlabeled_batch_size * unlabeled_data_utilization
        )
        print(f"effective_batch_size = {labeled_batch_size} + {unlabeled_batch_size} * {unlabeled_data_utilization} = {effective_batch_size}")
        performance_metrics_for_logging = dict(
            train_loss=loss,
            unlabeled_data_utilization=unlabeled_data_utilization
        )
        self.log_dict(
            dictionary=performance_metrics_for_logging,
            batch_size=effective_batch_size,
            on_epoch=True,
            on_step=False,
            prog_bar=True
        )
        return loss

    # def on_train_epoch_end(self):
    #     print(f"Epoch: {self.current_epoch}")
    #     print(f"lr = {self.trainer.optimizers[0].param_groups[0]['lr']}")

    def on_validation_epoch_start(self):
        print("\non_validation_epoch_start")
        self.accumulated_batches = defaultdict(list)

    def validation_step(self, batch, batch_idx):
        print(f"\nvalidation_step, {batch_idx =}")
        self._append_batch_to_accumulated_batches(batch)

    def on_validation_epoch_end(self):
        print("\non_validation_epoch_end")
        self._concatenate_accumulated_batches()

        logits = self._get_logits_by_image_and_augmentation_type(
            labeled_images=self.accumulated_batches['labeled_images'],
            unlabeled_images=None,
            training_step=False
        )

        performance_metrics_for_logging = dict(
            val_loss=self.criterion(
                logits=logits,
                labels=self.accumulated_batches['labels'],
                pipeline_step="validation"
            ),
            **self.performance_metrics.get(
                logits=logits,
                labels=self.accumulated_batches['labels'],
                subset_type_prefix="val"
            )
        )
        self.log_dict(
            dictionary=performance_metrics_for_logging,
            add_dataloader_idx=False,
            batch_size=self.accumulated_batches['labels'].shape[0],
            on_epoch=True,
            on_step=False,
            prog_bar=True
        )

    def on_test_epoch_start(self):
        print("\non_test_epoch_start")
        self.accumulated_batches = defaultdict(list)

    def test_step(self, batch, batch_idx):
        self._append_batch_to_accumulated_batches(batch)

    def on_test_epoch_end(self):
        self._concatenate_accumulated_batches()

        logits = self._get_logits_by_image_and_augmentation_type(
            labeled_images=self.accumulated_batches['labeled_images'],
            unlabeled_images=None,
            training_step=False
        )

        performance_metrics_for_logging = self.performance_metrics.get(
            logits=logits,
            labels=self.accumulated_batches['labels'],
            subset_type_prefix="test"
        )
        self.log_dict(
            dictionary=performance_metrics_for_logging,
            add_dataloader_idx=False,
            batch_size=self.accumulated_batches['labels'].shape[0],
            on_epoch=True,
            on_step=False,
            prog_bar=True
        )

    def _append_batch_to_accumulated_batches(self, batch):
        print("\n_append_batch_to_accumulated_batches")
        self.accumulated_batches[f'labeled_images'].append(
            batch[self.config.dataset_name.labeled]['data']['images']
        )
        self.accumulated_batches['labels'].append(
            batch[self.config.dataset_name.labeled]['labels']
        )
        print([
            labeled_images.shape[0]
            for labeled_images in self.accumulated_batches[f'labeled_images']
        ])
        print([
            labels.shape[0]
            for labels in self.accumulated_batches[f'labels']
        ])

    def _concatenate_accumulated_batches(self):
        print("\n_concatenate_accumulated_batches")
        self.accumulated_batches['labeled_images'] = torch.cat(
            self.accumulated_batches['labeled_images'],
            dim=0
        )
        self.accumulated_batches['labels'] = torch.cat(
            self.accumulated_batches['labels'],
            dim=0
        )
        print(f"{self.accumulated_batches['labeled_images'].shape = }")
        print(f"{self.accumulated_batches['labels'].shape = }")

    def _get_logits_by_image_and_augmentation_type(
            self,
            labeled_images,
            unlabeled_images,
            training_step
    ):
#        print("unlabeled_images is None?", unlabeled_images is None)
#        print("training_step?", training_step)

        logits = dict(
            from_non_augmented_labeled_images=None,
            from_weakly_augmented_labeled_images=None,
            from_weakly_augmented_unlabeled_images=None,
            from_strongly_augmented_unlabeled_images=None
        )

        if labeled_images is not None:
            weakly_augmented_labeled_images = \
                self.data_augmenter.get_weakly_augmented_images(
                    labeled_images
                )
            logits['from_non_augmented_labeled_images'] = \
                self(labeled_images)
            logits['from_weakly_augmented_labeled_images'] = \
                self(weakly_augmented_labeled_images)

        if training_step and unlabeled_images is not None:
            weakly_augmented_unlabeled_images = \
                self.data_augmenter.get_weakly_augmented_images(
                    unlabeled_images
                )
            strongly_augmented_unlabeled_images = \
                self.data_augmenter.get_strongly_augmented_images(
                    unlabeled_images
                )
            logits['from_weakly_augmented_unlabeled_images'] = \
                self(weakly_augmented_unlabeled_images)
            logits['from_strongly_augmented_unlabeled_images'] = \
                self(strongly_augmented_unlabeled_images)

        return logits

    def _setup(self, **kwargs):
        self.config = kwargs['config']
        self.data_augmenter = FixMatchDataAugmenter(
            config=self.config.data_augmenter
        )
        self.model = FixMatchModel(
            config=self.config.model
        )
        self.criterion = FixMatchLossFunction(
            config=self.config.criterion,
            experiment_execution_paths=kwargs['experiment_execution_paths']
        )
        self.performance_metrics = FixMatchPerformanceMetrics()
