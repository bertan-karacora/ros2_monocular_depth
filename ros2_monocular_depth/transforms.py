import torchvision.transforms.v2 as tv_transforms


MODES_INTERPOLATION = tv_transforms.InterpolationMode


def resize_shape_to_multiple_of_base(shape, min_size_resized, base):
    height, width = shape[2:]
    scale = min_size_resized / min(height, width)

    height_resized = round(scale * height / base) * base
    if height_resized < min_size_resized:
        height_resized = (int(scale * height / base) + 1) * base

    width_resized = round(scale * width / base) * base
    if width_resized < min_size_resized:
        width_resized = (int(scale * width / base) + 1) * base

    shape_resized = shape[:2] + (height_resized, width_resized)

    return shape_resized


class ResizeToMultipleOfBase(tv_transforms.Transform):
    def __init__(self, min_size_resized, base, mode_interpolation=MODES_INTERPOLATION.BILINEAR, use_antialiasing=True):
        super().__init__()

        self.base = base
        self.min_size_resized = min_size_resized
        self.mode_interpolation = mode_interpolation
        self.use_antialiasing = use_antialiasing

    def _transform(self, input, params):
        shape_resized = resize_shape_to_multiple_of_base(input.shape, min_size_resized=self.min_size_resized, base=self.base)
        output = tv_transforms.functional.resize(input, shape_resized[2:], interpolation=self.mode_interpolation, antialias=self.use_antialiasing)

        return output
