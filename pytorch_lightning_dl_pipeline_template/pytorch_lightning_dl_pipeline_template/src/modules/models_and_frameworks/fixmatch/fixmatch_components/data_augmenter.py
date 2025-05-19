from functools import wraps
import torch
import torchvision

from src.modules.models_and_frameworks.fixmatch.fixmatch_components \
    .data_augmenter_components \
    import FixMatchWeakDataAugmenter, FixMatchStrongDataAugmenter
# from src.modules.models_and_frameworks.fixmatch.fixmatch_components.data_transformations_3d import *

def enforce_argument_constraints():
    def decorator(method):
        @wraps(method)
        def wrapper(self, image):

            # Validate 'image' type
            actual_type = type(image)
            expected_type = torch.Tensor
            assert actual_type == expected_type, \
                f"Expected type {expected_type} for " \
                f"'image', got {actual_type}."

            # Validate 'image' shape
            actual_shape = image.shape[2:]
            expected_shape = torch.Size(self.config.data_size)
            assert actual_shape == expected_shape, \
                f"Expected shape {expected_shape} for " \
                f"'image', got {actual_shape}."

            return method(self, image)
        return wrapper
    return decorator

class FixMatchDataAugmenter:
    def __init__(self, config):
        self.config = config

        self.weak_data_augmenter = FixMatchWeakDataAugmenter(
            config=self.config.weak_data_augmenter
        )
        self.strong_data_augmenter = FixMatchStrongDataAugmenter(
            config=self.config.strong_data_augmenter
        )

    @enforce_argument_constraints()
    def get_weakly_augmented_images(self, image):
        transformed_image = \
            self.weak_data_augmenter.get_augmented_data(image)
        return transformed_image

    @enforce_argument_constraints()
    def get_strongly_augmented_images(self, image):
        transformed_image = \
            self.strong_data_augmenter.get_augmented_data(image)
        return transformed_image

    def _get_strong_data_transform_composition(self):
        if len(self.config.data_size) == 2:
            torchvision.transforms.Compose([
                lambda batch_of_imgs: torch.squeeze(batch_of_imgs, dim=1).cpu(),
                RandAugment(
                    number_of_augmentations=2,
                    volume_shape=hparams.volume_shape
                ),
                # RandomCutout(max_img_fraction=0.5),  # This augmentation was not used because it could remove a nodule and harm the classification.
                lambda batch_of_imgs: torch.unsqueeze(batch_of_imgs, dim=1).to(
                    "cuda:0" if torch.cuda.is_available() else "cpu")])
        elif len(self.config.data_size) == 3:
            torchvision.transforms.Compose([
                lambda batch_of_imgs: torch.squeeze(batch_of_imgs,
                                                    dim=1).cpu(),
                RandAugment3D(number_of_augmentations=2,
                              volume_shape=hparams.volume_shape,
                              verbose=verbose,
                              test=test),
                # RandomCutout(max_img_fraction=0.5),  # This augmentation was not used because it could remove a nodule and harm the classification.
                lambda batch_of_imgs: torch.unsqueeze(batch_of_imgs, dim=1).to(
                    "cuda:0" if torch.cuda.is_available() else "cpu")])
        else:
            raise ValueError(
                    f"Invalid data dimension: {len(self.config.data_size)}D. "
                    f"Supported data dimensions are '2D' or '3D'."
                )
        return weak_data_transform_composition

