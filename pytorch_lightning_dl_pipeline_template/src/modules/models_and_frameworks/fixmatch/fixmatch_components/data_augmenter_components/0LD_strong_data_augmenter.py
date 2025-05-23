import math
import monai
import numpy
import random
import torch
import torchio
import torchvision


class Rescale:
    def __call__(self, img, input_range, output_range):
        return (output_range[1] - output_range[0]) * (img - input_range[0]) / (input_range[1] - input_range[0]) + output_range[0]


class RandAugment3D:
    def __init__(self, augmentation_intensity, number_of_augmentations):
        self.augmentation_intensity = augmentation_intensity
        self.number_of_augmentations = number_of_augmentations

    def __call__(self, img):
        if self.augmentation_intensity == "random":
            augmentation_intensity = random.random()
        else:
            augmentation_intensity = self.augmentation_intensity
        rand_augmenter = torchvision.transforms.Compose(
            random.sample(population=self.data_transforms(augmentation_intensity), k=self.number_of_augmentations))
        return rand_augmenter(img)

    @staticmethod
    def data_transforms(augmentation_intensity):
        return [
            AutoContrast(),
            RandomBrightness(),
            RandomContrast(),
            Equalize(),
            Identity(),
            RandomPosterize(reduction_factor=2),
            RandomRotate(angle=(30 * augmentation_intensity, 30 * augmentation_intensity, 30 * augmentation_intensity)),
            RandomShear(x=0.3 * augmentation_intensity),
            RandomShear(y=0.3 * augmentation_intensity),
            RandomTranslate(x=0.3 * augmentation_intensity),
            RandomTranslate(y=0.3 * augmentation_intensity)]


class CutoutImageAugmentation:
    def __init__(self, number_of_holes=1, max_img_fraction=0.5, fill_value=1.0, p=1.):
        self.number_of_holes = number_of_holes
        self.max_img_fraction = max_img_fraction
        self.fill_value = fill_value
        self.p = p

    def __call__(self, img):
        if numpy.random.uniform() <= self.p:
            img_height = img.size()[2]
            img_width = img.size()[3]

            raw_mask = numpy.ones((img_height, img_width), numpy.float32)
            for n in range(self.number_of_holes):
                x = numpy.random.randint(img_width)
                y = numpy.random.randint(img_height)

                if isinstance(self.max_img_fraction, tuple):
                    img_height_fraction = numpy.random.uniform(0, self.max_img_fraction[0])
                    img_width_fraction = numpy.random.uniform(0, self.max_img_fraction[1])
                else:
                    img_height_fraction = img_width_fraction = numpy.random.uniform(0, self.max_img_fraction)

                x1 = numpy.clip(x - int(img_width_fraction * img_width // 2), 0, img_width)
                x2 = numpy.clip(x + int(img_width_fraction * img_width // 2), 0, img_width)
                y1 = numpy.clip(y - int(img_height_fraction * img_height // 2), 0, img_height)
                y2 = numpy.clip(y + int(img_height_fraction * img_height // 2), 0, img_height)

                raw_mask[y1: y2, x1: x2] = self.fill_value
            mask = torch.from_numpy(raw_mask).to(img.device)

            img = img * mask.expand_as(img)
        return img


class AutoContrast:
    def __call__(self, img):
        return Rescale(img, input_range=(img.min(), img.max()), output_range=(-1, img.max()))


class RandomBrightness:
    def __call__(self, img):
        return torchvision.transforms.ColorJitter(brightness=0.95, contrast=0.0, saturation=0, hue=0)


class RandomContrast:
    def __call__(self, img):
        return torchvision.transforms.ColorJitter(brightness=0.0, contrast=0.95, saturation=0, hue=0)


class Equalize:
    def __call__(self, img):
        self.equalize = torch.tensor(monai.transforms.utils.equalize_hist(img=img.numpy()))


class Identity:
    def __call__(self, img):
        return img


class RandomPosterize:
    def __init__(self, reduction_factor):
        self.reduction_factor = reduction_factor

    def __call__(self, img):
        return (img / self.reduction_factor).type(torch.LongTensor)


class RandomRotate:
    def __init__(self, angle=(0, 0, 0)):
        self.random_rotate = monai.transforms.RandRotate(
            range_x=math.radians(angle[0]), range_y=math.radians(angle[0]), range_z=math.radians(angle[0]), prob=1.0)

    def __call__(self, img):
        return self.random_rotate(img)


class RandomShear:
    def __init__(self, x=0.0, y=0.0):
        self.random_translate = torchvision.transforms.RandomAffine(degrees=0, shear=(x, y))

    def __call__(self, img):
        rescaled_img = Rescale(img, input_range=(img.min(), img.max()), output_range=(0, img.max()))
        sheared_rescaled_img = self.random_translate(rescaled_img)
        sheared_img = Rescale(
            img=sheared_rescaled_img, input_range=(sheared_rescaled_img.min(), sheared_rescaled_img.max()),
            output_range=(img.min(), sheared_rescaled_img.max()))
        return sheared_img


class RandomTranslate:
    def __init__(self, x=0.0, y=0.0):
        self.random_translate = torchvision.transforms.RandomAffine(degrees=0, translate=(x, y))

    def __call__(self, img):
        rescaled_img = Rescale(img, input_range=(img.min(), img.max()), output_range=(0, img.max()))
        translated_rescaled_img = self.random_translate(rescaled_img)
        translated_img = Rescale(
            img=translated_rescaled_img, input_range=(translated_rescaled_img.min(), translated_rescaled_img.max()),
            output_range=(img.min(), translated_rescaled_img.max()))
        return translated_img
