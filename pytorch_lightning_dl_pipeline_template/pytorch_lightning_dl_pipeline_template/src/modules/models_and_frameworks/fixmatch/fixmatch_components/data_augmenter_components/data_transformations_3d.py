import math
import monai
import numpy
import random
import torch
import torchio
import torchvision


def rescale(img, input_range, output_range):
    return (output_range[1] - output_range[0]) * (img - input_range[0]) / (input_range[1] - input_range[0]) + output_range[0]


class AutoContrast:
    def __call__(self, img):
        return rescale(img, input_range=(img.min(), img.max()), output_range=(-1, img.max()))


class Equalize:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, img):
        if self.verbose:
            print("Equalize called")
        return torch.tensor(monai.transforms.utils.equalize_hist(img=img.numpy(), min=-1, max=1))


class Identity:
    def __init__(self, verbose=False):
        self.verbose = verbose

    def __call__(self, img):
        if self.verbose:
            print("Identity called")
        return img


class RandAugment3D:
    def __init__(self, number_of_augmentations, volume_shape, verbose=False):
        self.number_of_augmentations = number_of_augmentations
        self.test = test
        self.verbose = verbose
        self.volume_shape = volume_shape

    def __call__(self, img):
        rand_augmenter = torchvision.transforms.Compose(
            random.sample(
                population=self.data_transforms(),
                k=self.number_of_augmentations
            )
        )
        return rand_augmenter(img)

    def data_transforms(self):
        return [
            # RandomBlur(gaussian_kernel_std=(0, 0.5)),
            # RandomBrightness(intensity=0.25),
            # RandomContrast(intensity=0.25),
            Equalize(verbose=self.verbose),
            Identity(verbose=self.verbose),
            RandomRotate(angles=(15, 0, 0), verbose=self.verbose, test=self.test),
            RandomRotate(angles=(0, 15, 0), verbose=self.verbose, test=self.test),
            RandomRotate(angles=(0, 0, 15), verbose=self.verbose, test=self.test),
            RandomShear(shear_range=(15, 0, 0), verbose=self.verbose, test=self.test),
            RandomShear(shear_range=(0, 15, 0), verbose=self.verbose, test=self.test),
            RandomShear(shear_range=(0, 0, 15), verbose=self.verbose, test=self.test),
            RandomTranslate(translation=(round(0.125 * self.volume_shape[-3]), 0, 0), verbose=self.verbose, test=self.test),
            RandomTranslate(translation=(0, round(0.125 * self.volume_shape[-2]), 0), verbose=self.verbose, test=self.test),
            RandomTranslate(translation=(0, 0, round(0.125 * self.volume_shape[-1])), verbose=self.verbose, test=self.test)]


class RandomBlur:
    def __init__(self, gaussian_kernel_std=(0, 0)):
        self.random_blurrer = torchio.transforms.RandomBlur(std=gaussian_kernel_std)

    def __call__(self, img):
        print("RandomBlur called")
        return self.random_blurrer(img)


class RandomBrightness:
    def __init__(self, intensity=0.0):
        self.intensity = 2 * intensity

    def __call__(self, img):
        print("RandomBrightness called")
        brightness_adjustment_factor = random.uniform(a=-self.intensity, b=self.intensity)
        brightness_adjusted_img = img + brightness_adjustment_factor
        return torch.maximum(torch.minimum(brightness_adjusted_img, torch.full(img.shape, fill_value=1)), torch.full(img.shape, fill_value=-1))


class RandomContrast:
    def __init__(self, intensity=0.0):
        self.intensity = intensity

    def __call__(self, img):
        print("RandomContrast called")
        raw_contrast_adjustment_factor = random.uniform(a=-self.intensity, b=self.intensity)
        contrast_adjustment_factor = raw_contrast_adjustment_factor + 1
        contrast_adjusted_img = img * contrast_adjustment_factor
        return torch.maximum(torch.minimum(contrast_adjusted_img, torch.full(img.shape, fill_value=1)), torch.full(img.shape, fill_value=-1))


class RandomCutout:
    def __init__(self, number_of_holes=1, max_img_fraction=0, fill_value=1.0, p=1., verbose=False):
        self.number_of_holes = number_of_holes
        self.max_img_fraction = max_img_fraction
        self.fill_value = fill_value
        self.p = p
        self.verbose = verbose

    def __call__(self, bath_of_imgs):
        if self.verbose:
            print(f"RandomCutout called with number_of_holes = {self.number_of_holes}, "
                  f"max_img_fraction = {self.max_img_fraction}, fill_value = {self.fill_value}")
        for img_index in range(len(bath_of_imgs)):
            if numpy.random.uniform() <= self.p:
                img_dim_0_length = bath_of_imgs[img_index].shape[-3]
                img_dim_1_length = bath_of_imgs[img_index].shape[-2]
                img_dim_2_length = bath_of_imgs[img_index].shape[-1]

                for n in range(self.number_of_holes):
                    img_dim_0_value = numpy.random.randint(img_dim_0_length)
                    img_dim_1_value = numpy.random.randint(img_dim_1_length)
                    img_dim_2_value = numpy.random.randint(img_dim_2_length)
                    if isinstance(self.max_img_fraction, tuple):
                        img_dim_0_length_fraction = numpy.random.uniform(0, self.max_img_fraction[0])
                        img_dim_1_length_fraction = numpy.random.uniform(0, self.max_img_fraction[1])
                        img_dim_2_length_fraction = numpy.random.uniform(0, self.max_img_fraction[2])
                    else:
                        img_dim_0_length_fraction = numpy.random.uniform(0, self.max_img_fraction)
                        img_dim_1_length_fraction = numpy.random.uniform(0, self.max_img_fraction)
                        img_dim_2_length_fraction = numpy.random.uniform(0, self.max_img_fraction)

                    img_dim_0_value_1 = numpy.clip(img_dim_0_value - int(img_dim_0_length_fraction * img_dim_0_length // 2), 0, img_dim_0_length)
                    img_dim_0_value_2 = numpy.clip(img_dim_0_value + int(img_dim_0_length_fraction * img_dim_0_length // 2), 0, img_dim_0_length)
                    img_dim_1_value_1 = numpy.clip(img_dim_1_value - int(img_dim_1_length_fraction * img_dim_1_length // 2), 0, img_dim_1_length)
                    img_dim_1_value_2 = numpy.clip(img_dim_1_value + int(img_dim_1_length_fraction * img_dim_1_length // 2), 0, img_dim_1_length)
                    img_dim_2_value_1 = numpy.clip(img_dim_2_value - int(img_dim_2_length_fraction * img_dim_2_length // 2), 0, img_dim_2_length)
                    img_dim_2_value_2 = numpy.clip(img_dim_2_value + int(img_dim_2_length_fraction * img_dim_2_length // 2), 0, img_dim_2_length)
                    bath_of_imgs[img_index][
                        img_dim_0_value_1: img_dim_0_value_2, img_dim_1_value_1: img_dim_1_value_2, img_dim_2_value_1: img_dim_2_value_2] = self.fill_value
        return bath_of_imgs


class RandomRotate:
    def __init__(self, angles=(0, 0, 0), test=False, verbose=False):
        self.angles = angles
        self.random_rotate = monai.transforms.RandRotate(
            padding_mode="zeros", range_x=math.radians(angles[0]), range_y=math.radians(angles[1]), range_z=math.radians(angles[2]), prob=1.0)
        self.test = test
        self.verbose = verbose

    def __call__(self, volume):
        if self.verbose:
            print(f"RandomRotate called with angles = {self.angles}")
        rescaled_volume = rescale(volume, input_range=(volume.min(), volume.max()), output_range=(volume.min() if self.test else 0, volume.max()))
        rotated_rescaled_volume = self.random_rotate(rescaled_volume)
        rotated_volume = rescale(
            img=rotated_rescaled_volume, input_range=(rotated_rescaled_volume.min(), rotated_rescaled_volume.max()),
            output_range=(volume.min(), volume.max()))
        return rotated_volume


class RandomShear:
    def __init__(self, shear_range, verbose=False, test=False):
        self.random_translate = monai.transforms.RandAffine(
            padding_mode="zeros", prob=1.0, shear_range=tuple([0, *[math.radians(axis_shear_range) for axis_shear_range in shear_range]]))
        self.shear_range = shear_range
        self.test = test
        self.verbose = verbose

    def __call__(self, volume):
        if self.verbose:
            print(f"RandomShear called with shear_range = {self.shear_range}")
        rescaled_volume = rescale(volume, input_range=(volume.min(), volume.max()), output_range=(volume.min() if self.test else 0, volume.max()))
        sheared_rescaled_volume = self.random_translate(rescaled_volume)
        sheared_volume = rescale(
            img=sheared_rescaled_volume, input_range=(sheared_rescaled_volume.min(), sheared_rescaled_volume.max()),
            output_range=(volume.min(), volume.max()))
        return sheared_volume


class RandomTranslate:
    def __init__(self, translation, verbose=False, test=False):
        default_pad_value = 0 if test else -1
        self.random_translate = torchio.transforms.RandomAffine(scales=0, degrees=0, translation=translation, default_pad_value=default_pad_value)
        self.translation = translation
        self.verbose = verbose

    def __call__(self, volume):
        if self.verbose:
            print(f"RandomTranslate called with translation = {self.translation}")
        return self.random_translate(volume)
