import torch
import torchio
import torchvision


class FixMatchWeakDataAugmenter:
    def __init__(self, config):
        self.config = config

        self.weak_data_transform_composition = \
            self._get_weak_data_transform_composition()

    def get_augmented_data(self, images):
        transformed_image = torch.stack([
            self.weak_data_transform_composition(image) for image in images
        ])
        return transformed_image

    def _get_weak_data_transform_composition(self):
        if len(self.config.data_size) == 2:
            weak_data_transform_composition = torchvision.transforms.Compose([
                torchvision.transforms.RandomHorizontalFlip(
                    p=self.config.random_flip.probability
                ),
                torchvision.transforms.RandomVerticalFlip(
                    p=self.config.random_flip.probability
                ),
                torchvision.transforms.RandomAffine(
                    degrees=0,
                    translate=(
                        self.config.random_translate.maximum_fraction,
                        self.config.random_translate.maximum_fraction
                    ),
                    fill=self.config.random_translate.fill
                )
            ])
        elif len(self.config.data_size) == 3:
            # TODO
            weak_data_transform_composition = torchvision.transforms.Compose([
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
            ])
        else:
            raise ValueError(
                    f"Invalid data dimension: {len(self.config.data_size)}D. "
                    f"Supported data dimensions are '2D' or '3D'."
                )
        return weak_data_transform_composition
