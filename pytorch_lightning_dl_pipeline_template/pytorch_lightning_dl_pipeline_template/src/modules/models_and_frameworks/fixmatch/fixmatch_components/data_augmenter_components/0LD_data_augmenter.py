import torch
import torchio
import torchvision

from modules.fixmatch_components.data_transformations_3d import *


class DataAugmenter:
    def __init__(self, hparams, verbose=False, test=False):
        self._hparams = hparams
        self.weak_data_transform_composition = torchvision.transforms.Compose([
            lambda batch_of_imgs: torch.squeeze(batch_of_imgs, dim=1).cpu(),
            torchio.transforms.RandomFlip(axes=(0, 1, 2)),
            RandomTranslate(
                translation=(round(0.125 * hparams.volume_shape[-3]), round(0.125 * hparams.volume_shape[-2]), round(0.125 * hparams.volume_shape[-1])),
                verbose=verbose, test=test),
            lambda batch_of_imgs: torch.unsqueeze(batch_of_imgs, dim=1).to("cuda:0" if torch.cuda.is_available() else "cpu")])
        self.strong_data_transform_composition = torchvision.transforms.Compose([
            lambda batch_of_imgs: torch.squeeze(batch_of_imgs, dim=1).cpu(),
            RandAugment3D(number_of_augmentations=2, volume_shape=hparams.volume_shape, verbose=verbose, test=test),
            # RandomCutout(max_img_fraction=0.5),  # This augmentation was not used because it could remove a nodule and harm the classification.
            lambda batch_of_imgs: torch.unsqueeze(batch_of_imgs, dim=1).to("cuda:0" if torch.cuda.is_available() else "cpu")])

    def __call__(self, batch_of_imgs, intensity):
        if intensity == "weak":
            transformed_img = self.weak_data_transform_composition(batch_of_imgs)
        elif intensity == "strong":
            transformed_img = self.strong_data_transform_composition(batch_of_imgs)
        else:
            raise ValueError("Invalid intensity")
        return transformed_img
