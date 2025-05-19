from monai.transforms import RandAffine
from torchvision.transforms.v2 import (
    ColorJitter,
    Compose,
    ElasticTransform,
    GaussianBlur,
    Lambda,
    Normalize,
    RandomAffine,
    RandomErasing,
    RandomRotation
)
import random
import torch
import torchio



class FixMatchStrongDataAugmenter:
    def __init__(self, config):
        self.config = config

        self.randaugment_transforms = self._get_randaugment_transforms()
        self.random_cutout_transform = self._get_random_cutout_transform()

    def get_augmented_data(self, images):
        transformed_image = torch.stack([
            self._get_transformed_image(image) for image in images
        ])
        return transformed_image

    def _get_random_cutout_transform(self):
        random_cutout_transform = Compose([*[
            RandomErasing(
                p=1.0,
                scale=tuple(self.config.random_erasing.scale),
                ratio=tuple(self.config.random_erasing.ratio),
                value=self.config.random_erasing.value
            ) for _ in range(10)
        ]])
        return random_cutout_transform

    def _get_transformed_image(self, image):
        random_strong_data_transforms = random.sample(
            self.randaugment_transforms,
            k=self.config.number_of_transformations
        )
        strong_data_transform_composition = \
            Compose(random_strong_data_transforms)
        transformed_image = self.random_cutout_transform(
            strong_data_transform_composition(image)
        )
        return transformed_image

    def _get_randaugment_transforms(self):
        if len(self.config.data_size) == 2:
            strong_data_transforms = [
                ElasticTransform(
                    alpha=self.config.random_elastic_transform.alpha,
                    sigma=self.config.random_elastic_transform.sigma,
                    fill=self.config.random_elastic_transform.fill
                ),
                RandomAffine(
                    degrees=0,
                    translate=(
                        self.config.random_translate.maximum_fraction,
                        self.config.random_translate.maximum_fraction
                    ),
                    fill=self.config.random_translate.fill
                ),
                Compose([
                    Normalize(mean=[-1.0], std=[2.0]),
                    RandAffine(
                        padding_mode="zeros",
                        prob=1.0,
                        shear_range=tuple([
                            0, self.config.random_shear.fraction_range
                        ])
                    ),
                    Normalize(mean=[0.5], std=[0.5])
                ]),
                Compose([
                    Normalize(mean=[-1.0], std=[2.0]),
                    RandAffine(
                        padding_mode="zeros",
                        prob=1.0,
                        shear_range=tuple([
                            self.config.random_shear.fraction_range, 0
                        ])
                    ),
                    Normalize(mean=[0.5], std=[0.5])
                ]),
                Lambda(lambda image: image),
                RandomRotation(
                    degrees=self.config.random_rotation.degrees,
                    fill=self.config.random_rotation.fill
                ),
                GaussianBlur(
                    kernel_size=self.config.random_gaussian_blur.kernel_size,
                    sigma=self.config.random_gaussian_blur.sigma_range
                ),
                Compose([
                    Normalize(mean=[-1.0], std=[2.0]),
                    ColorJitter(
                        brightness=
                            tuple(self.config.random_brightness.factor_range)
                    ),
                    Normalize(mean=[0.5], std=[0.5])
                ]),
                Compose([
                    Normalize(mean=[-1.0], std=[2.0]),
                    ColorJitter(
                        contrast=tuple(self.config.random_contrast.factor_range)
                    ),
                    Normalize(mean=[0.5], std=[0.5])
                ])
            ]
        elif len(self.config.data_size) == 3:
            # TODO
            strong_data_transforms = [
                lambda batch_of_imgs: torch.squeeze(batch_of_imgs, dim=1).cpu(),
                torchio.transforms.RandomFlip(axes=(0, 1, 2)),
                RandomTranslate(
                    translation=(
                        round(0.125 * self.config.data_size[-3]),
                        round(0.125 * self.config.data_size[-2]),
                        round(0.125 * self.config.data_size[-1])
                    )
                ),
                lambda batch_of_imgs: torch.unsqueeze(batch_of_imgs, dim=1).to(
                    "cuda:0" if torch.cuda.is_available() else "cpu")
            ]
        else:
            raise ValueError(
                    f"Invalid data dimension: {len(self.config.data_size)}D. "
                    f"Supported data dimensions are '2D' or '3D'."
                )
        return strong_data_transforms
