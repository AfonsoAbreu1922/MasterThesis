from collections import defaultdict
from pytorch_lightning.utilities import CombinedLoader
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import DataLoader as TorchDataLoader
import numpy
import random
import torch

from src.modules.data.dataloader.preprocessed_dataloaders \
    .lidc_idri_preprocessed_dataloader \
    import LIDCIDRIPreprocessedDataLoader
from src.modules.data.dataloader.preprocessed_dataloaders \
    .luna25_preprocessed_dataloader \
    import LUNA25PreprocessedDataLoader


class CombinedPreprocessedKFoldDataLoader:
    def __init__(self, **kwargs):
        self.config = None
        self.dataloaders = None
        self.data_splits = None
        self.preprocessed_dataloader_class = None
        self.torch_generator = None

        self._setup(**kwargs)

    def get_data_names(self):
        data_names = {subset_type: [
            self.data_splits[subset_type]['file_names'][datafold_id]
            for datafold_id in range(self.config.data_split.number_of_k_folds)
        ] for subset_type in ["training", "validation", "test"]}
        return data_names

    def get_dataloaders(self):
        return self.dataloaders

    def _get_torch_dataloader(
            self,
            dataset_name,
            file_names,
            labels,
            subset_type,
            torch_dataloader_kwargs
    ):
        torch_dataloader = TorchDataLoader(
            dataset=self.preprocessed_dataloader_class[dataset_name](
                config=self.config.dataloaders[dataset_name],
                file_names=file_names,
                labels=labels,
                subset_type=subset_type
            ),
            generator=self.torch_generator,
            shuffle=True if subset_type == "train" else False,
            worker_init_fn=self._get_torch_dataloader_worker_init_fn,
            **torch_dataloader_kwargs
        )
        return torch_dataloader

    def _get_torch_dataloader_worker_init_fn(self, worker_id):
        numpy.random.seed(self.config.seed_value + worker_id)
        random.seed(self.config.seed_value + worker_id)

    def _set_dataloaders(self):
        self.dataloaders = defaultdict(list)
        for subset_type in ["training", "validation", "test"]:
            for datafold_id in range(
                    self.config.data_split.number_of_k_folds
            ):
                dataloaders = {
                    dataset_name: self._get_torch_dataloader(
                        dataset_name=dataset_name,
                        file_names=
                            self.data_splits[dataset_name][subset_type] \
                            ['file_names'][datafold_id],
                        labels=
                            self.data_splits[dataset_name][subset_type] \
                            ['labels'][datafold_id],
                        subset_type=subset_type,
                        torch_dataloader_kwargs=
                            self.config.dataloaders[dataset_name]
                            .torch_dataloader_kwargs
                    ) for dataset_name
                    in self.config.dataloader_split[subset_type]
                }
                self.dataloaders[subset_type].append(
                    CombinedLoader(
                        iterables=dataloaders,
                        mode=self.config.mode
                            # if subset_type == "train" else "sequential"
                    )
                )

    def _set_data_splits(self, lung_nodule_metadataframes):

        def append_data_to_data_splits_attribute():
            self.data_splits[dataset.name]['training']['file_names'].append(
                train_lung_nodule_metadataframe['file_name'].tolist()
            )
            self.data_splits[dataset.name]['training']['labels'].append(
                train_lung_nodule_metadataframe
                [stratification_label_column_name].tolist()
            )
            self.data_splits[dataset.name]['validation']['file_names'].append(
                validation_lung_nodule_metadataframe['file_name'].tolist()
            )
            self.data_splits[dataset.name]['validation']['labels'].append(
                validation_lung_nodule_metadataframe
                [stratification_label_column_name].tolist()
            )
            self.data_splits[dataset.name]['test']['file_names'].append(
                test_lung_nodule_metadataframe['file_name'].tolist()
            )
            self.data_splits[dataset.name]['test']['labels'].append(
                test_lung_nodule_metadataframe
                [stratification_label_column_name].tolist()
            )

        self.data_splits = defaultdict(
            lambda: defaultdict(lambda: defaultdict(list))
        )
        if self.config.data_split.number_of_k_folds == 1:
            for dataset in self.config.dataset:
                stratification_label_column_name = 'label_{}'.format(
                    self.config.data_split.stratify_label[dataset.name]
                )

                (
                    train_and_validation_lung_nodule_metadataframe,
                    test_lung_nodule_metadataframe
                ) = train_test_split(
                    lung_nodule_metadataframes[dataset.name],
                    test_size=self.config.data_split
                        .test_fraction_of_entire_dataset,
                    random_state=self.config.seed_value,
                    stratify=lung_nodule_metadataframes[dataset.name]
                        [stratification_label_column_name]
                )

                (
                    train_lung_nodule_metadataframe,
                    validation_lung_nodule_metadataframe
                ) = train_test_split(
                    train_and_validation_lung_nodule_metadataframe,
                    test_size=self.config.data_split
                        .validation_fraction_of_train_set,
                    random_state=self.config.seed_value,
                    stratify=train_and_validation_lung_nodule_metadataframe
                        [stratification_label_column_name]
                )

                append_data_to_data_splits_attribute()
        else:
            for dataset in self.config.dataset:
                skf_cross_validator = StratifiedKFold(
                    n_splits=
                        self.config.data_split.number_of_k_folds,
                    shuffle=True,
                    random_state=self.config.seed_value
                )
                stratification_label_column_name = 'label_{}'.format(
                    self.config.data_split.stratify_label[dataset.name]
                )

                skf_split_generator = skf_cross_validator.split(
                    X=lung_nodule_metadataframes[dataset.name],
                    y=lung_nodule_metadataframes[dataset.name]
                        [stratification_label_column_name]
                )

                for datafold_id, (train_and_validation_indexes, test_indexes) \
                        in enumerate(skf_split_generator, 1):
                    test_lung_nodule_metadataframe = \
                        lung_nodule_metadataframes \
                            [dataset.name].iloc[test_indexes]
                    (
                        train_lung_nodule_metadataframe,
                        validation_lung_nodule_metadataframe
                    ) = train_test_split(
                        lung_nodule_metadataframes[dataset.name] \
                            .iloc[train_and_validation_indexes],
                        test_size=self.config.data_split
                            .validation_fraction_of_train_set,
                        random_state=self.config.seed_value,
                        stratify=lung_nodule_metadataframes[dataset.name]
                            [stratification_label_column_name]
                            .iloc[train_and_validation_indexes]
                    )

                    append_data_to_data_splits_attribute()

    def _setup(self, **kwargs):
        self.config = kwargs['config']
        self.preprocessed_dataloader_class = dict(
            lidc_idri=LIDCIDRIPreprocessedDataLoader,
            luna25=LUNA25PreprocessedDataLoader
        )
        self.torch_generator = torch.Generator()
        self.torch_generator.manual_seed(kwargs['config'].seed_value)
        self._set_data_splits(kwargs['lung_nodule_metadataframes'])
        self._set_dataloaders()