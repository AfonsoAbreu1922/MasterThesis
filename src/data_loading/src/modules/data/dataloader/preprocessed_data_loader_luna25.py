import torch
import numpy as np
from torch.utils.data import Dataset
import torchvision

from src.modules.data.data_augmentation import CTImageAugmenter  # Optional, based on my structure

class LUNA25PreprocessedDataLoader(Dataset):
    def __init__(
        self,
        config,
        file_names,
        image_numpy_arrays_dir_path,
        label_dataframe,
        subset_type
    ):
        self.config = config
        self.file_names = file_names
        self.label_dataframe = label_dataframe
        self.subset_type = subset_type

        self.apply_data_augmentations = config.data_augmentation.apply
        if self.apply_data_augmentations and subset_type == "train":
            self.data_augmenter = CTImageAugmenter(parameters=config.data_augmentation.parameters)

        self.image_numpy_arrays_dir_path = image_numpy_arrays_dir_path
        self.image_transformer = torchvision.transforms.Compose([
            lambda x: np.transpose(x, axes=(1, 2, 0)) if x.ndim == 3 else x,
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize(mean=0.5, std=0.5),
        ])

    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, data_index):
        image = np.load(f"{self.image_numpy_arrays_dir_path}/{self.file_names[data_index]}.npy").astype(np.float32)

        if self.apply_data_augmentations and self.subset_type == "train":
            image = self.data_augmenter(image=image)

        transformed_image = self.image_transformer(image)
        label = torch.tensor([
            self.label_dataframe.loc[
                self.label_dataframe['file_name'] == self.file_names[data_index],
                'label'
            ].values[0]
        ])

        return {"input_image": transformed_image}, {"lnm": {"mean": label}}
