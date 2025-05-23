import numpy
import torch


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
