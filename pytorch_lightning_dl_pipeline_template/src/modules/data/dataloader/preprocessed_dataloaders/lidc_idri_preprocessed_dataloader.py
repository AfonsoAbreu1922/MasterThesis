from collections import defaultdict
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch.utils.data import Dataset
from torch.utils.data import DataLoader as TorchDataLoader
import numpy
import random
import torch
import torchvision

from src.modules.data.data_augmentation.ct_image_augmenter \
    import CTImageAugmenter


class LIDCIDRIPreprocessedKFoldDataLoader:
    def __init__(self, **kwargs):
        self.config = None
        self.dataloaders = None
        self.dataloaders_by_subset = None
        self.data_names_by_subset = None
        self.data_splits = None
        self.torch_generator = None

        self._setup(**kwargs)

    def get_data_names(self):
        data_names = {subset_type: [
            self.data_splits[subset_type]['file_names'][datafold_id]
            for datafold_id in range(self.config.number_of_k_folds)
        ] for subset_type in ["training", "validation", "test"]}
        return data_names

    def get_dataloaders(self):
        return self.dataloaders

    def _get_torch_dataloader(
            self,
            file_names,
            labels,
            subset_type,
            torch_dataloader_kwargs
    ):
        torch_dataloader = TorchDataLoader(
            dataset=LIDCIDRIPreprocessedDataLoader(
                config=self.config,
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
        for subset_type in ["train", "validation", "test"]:
            for datafold_id in range(self.config.number_of_k_folds):
                self.dataloaders[subset_type].append(
                    self._get_torch_dataloader(
                        file_names=self.data_splits[subset_type] \
                            ['file_names'][datafold_id],
                        labels=self.data_splits[subset_type] \
                            ['labels'][datafold_id],
                        subset_type=subset_type,
                        torch_dataloader_kwargs=
                            self.config.torch_dataloader_kwargs
                    )
                )

    def _set_data_splits(self, lung_nodule_metadataframe):
        self.data_splits = defaultdict(lambda: defaultdict(list))
        if not self.config.number_of_k_folds:
            train_and_validation_file_name_column, test_file_name_column = \
                train_test_split(
                    lung_nodule_metadataframe,
                    test_size=self.config.test_fraction_of_entire_dataset,
                    random_state=self.config.seed_value,
                    stratify=lung_nodule_metadataframe['label']
                )
            train_file_name_column, validation_file_name_column = \
                train_test_split(
                    train_and_validation_file_name_column,
                    test_size=self.config.validation_fraction_of_train_set,
                    random_state=self.config.seed_value,
                    stratify=lung_nodule_image_metadataframe[
                        lung_nodule_image_metadataframe['file_name'].isin(
                            train_and_validation_file_name_column
                        )
                    ]['Mean Nodule Malignancy'].apply(lambda x: int(x + 0.5))
                )

            self.data_names_by_subset['train'] = \
                train_file_name_column.tolist()
            self.data_names_by_subset['validation'] = \
                validation_file_name_column.tolist()
            self.data_names_by_subset['test'] = \
                test_file_name_column.tolist()

            for subset_type in ["train", "validation", "test"]:
                self.dataloaders_by_subset[subset_type] = \
                    self._get_torch_dataloader(
                        file_names=self.data_names_by_subset[subset_type],
                        label_dataframe=lung_nodule_image_metadataframe,
                        subset_type=subset_type,
                        torch_dataloader_kwargs=
                            self.config.torch_dataloader_kwargs
                    )
        else:
            skf_cross_validator = StratifiedKFold(
                n_splits=self.config.number_of_k_folds,
                shuffle=True,
                random_state=self.config.seed_value
            )
            skf_split_generator = skf_cross_validator.split(
                X=lung_nodule_metadataframe,
                y=lung_nodule_metadataframe['label']
            )

            for datafold_id, (train_and_validation_indexes, test_indexes) \
                    in enumerate(skf_split_generator, 1):
                test_lung_nodule_metadataframe = \
                    lung_nodule_metadataframe.iloc[test_indexes]
                (
                    train_lung_nodule_metadataframe,
                    validation_lung_nodule_metadataframe
                ) = train_test_split(
                    lung_nodule_metadataframe \
                        .iloc[train_and_validation_indexes],
                    test_size=self.config.validation_fraction_of_train_set,
                    random_state=self.config.seed_value,
                    stratify=lung_nodule_metadataframe['label'] \
                        .iloc[train_and_validation_indexes]
                )

                self.data_splits['train']['file_names'].append(
                    train_lung_nodule_metadataframe['file_name'].tolist()
                )
                self.data_splits['train']['labels'].append(
                    train_lung_nodule_metadataframe['label'].tolist()
                )
                self.data_splits['validation']['file_names'].append(
                    validation_lung_nodule_metadataframe['file_name'].tolist()
                )
                self.data_splits['validation']['labels'].append(
                    validation_lung_nodule_metadataframe['label'].tolist()
                )
                self.data_splits['test']['file_names'].append(
                    test_lung_nodule_metadataframe['file_name'].tolist()
                )
                self.data_splits['test']['labels'].append(
                    test_lung_nodule_metadataframe['label'].tolist()
                )

    def _setup(self, **kwargs):
        self.config = kwargs['config']
        self.torch_generator = torch.Generator()

        self.torch_generator.manual_seed(kwargs['config'].seed_value)
        self._set_data_splits(kwargs['lung_nodule_metadataframe'])
        self._set_dataloaders()


class LIDCIDRIPreprocessedDataLoader(Dataset):
    def __init__(self, **kwargs):
        self.apply_data_augmentations = None
        self.config = None
        self.data_augmenter = None
        self.data_transformer = None
        self.file_names = None
        self.labels = None
        self.load_data = None
        self.load_labels = None
        self.load_metadata = None

        self._setup(**kwargs)

    def __len__(self):
        if self.apply_data_augmentations:
            return len(self.file_names['with_data_augmentation'])
        else:
            return len(self.file_names['without_data_augmentation'])

    def __getitem__(self, data_index):
        item = dict(data=self._get_data(data_index))
        if self.load_labels:
            item['labels'] = self._get_labels(data_index)
        if self.load_metadata['file_name']:
            item['file_names'] = self.file_names[data_index]
        return item

    def _get_data(self, data_index):
        data = {}
        if not self.load_data['mask']:
            if not self.apply_data_augmentations:
                image = numpy.load("{}/{}.npy".format(
                    self.config.paths.image_numpy_arrays_dir,
                    self.file_names['without_data_augmentation'][data_index]
                )).astype(numpy.float32)
            else:
                image = numpy.load("{}/{}.npy".format(
                    self.config.paths.image_numpy_arrays_dir,
                    self.file_names['with_data_augmentation'][data_index]
                )).astype(numpy.float32)
                if (
                        data_index
                        >= len(self.file_names['without_data_augmentation'])
                ):
                    image = self.data_augmenter.get_augmented_data(image=image)
            data['images'] = self.data_transformer['image'](image)
        elif self.load_data['mask']:
            image = numpy.load(
                f"{self.config.paths.image_numpy_arrays_dir}"
                f"/{self.file_names[data_index]}.npy"
            ).astype(numpy.float32)
            mask = numpy.load(
                f"{self.config.paths.mask_numpy_arrays_dir}"
                f"/{self.file_names[data_index]}.npy"
            ).astype(numpy.float32)
            if self.apply_data_augmentations:
                if data_index >= len(self.file_names['without_data_augmentation']):
                    image, mask = self.data_augmenter.get_augmented_data(
                        image=image,
                        mask=mask
                    )
            data['images'] = self.data_transformer['image'](image)
            data['masks'] = self.data_transformer['mask'](mask)
        # print(f"LIDC {self.file_names['without_data_augmentation'][data_index] = } {data['images'].shape = }")
        return data

    def _get_labels(self, data_index):
        labels = torch.tensor([
            float(self.labels[data_index])
        ])
        # print(f"LIDC {labels.shape = }")
        return labels

    def _setup(self, **kwargs):
        self.config = kwargs['config']

        self.data_transformer = dict(
            image=torchvision.transforms.Compose([
                lambda x: numpy.transpose(x, axes=(1, 2, 0))
                if x.ndim == 3 else x,
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(mean=0.5, std=0.5),
            ])
        )

        if kwargs['config'].load.data.mask:
            self.data_transformer['mask'] = torchvision.transforms.Compose([
                lambda x: numpy.transpose(x, axes=(1, 2, 0))
                if x.ndim == 3 else x,
                torchvision.transforms.ToTensor(),
            ])

        self.file_names = dict(without_data_augmentation=kwargs['file_names'])
        if (
                kwargs['config'].data.augmentation.apply
                and kwargs['subset_type'] == "train"
        ):
            self.apply_data_augmentations = True
            self.data_augmenter = CTImageAugmenter(
                parameters=kwargs['config'].data_augmentation.parameters
            )
            self.file_names['with_data_augmentation'] = (
                kwargs['file_names']
                + kwargs['config'].data.augmentation
                    .augmented_to_original_data_ratio
                * kwargs['file_names']
            )
        else:
            self.apply_data_augmentations = False

        self.labels = kwargs['labels']
        self.load_data = kwargs['config'].load.data
        self.load_labels = kwargs['config'].load.labels
        self.load_metadata = kwargs['config'].load.metadata