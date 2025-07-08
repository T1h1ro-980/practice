import random
import torchvision.transforms.functional as F
from torchvision.transforms import GaussianBlur
from torchvision.transforms.functional import to_tensor, to_pil_image

class RandomGaussianBlur:
    def __init__(self, p=0.5, kernel_size=(5, 5), sigma=(0.1, 2.0)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, img):
        if random.random() > self.p:
            return img
        
        blur = GaussianBlur(kernel_size=self.kernel_size, sigma=self.sigma)
        return blur(img)
   
   
   
import random
from torchvision.transforms import functional as F
from torchvision.transforms.functional import InterpolationMode

class RandomPerspectiveTransform:
    def __init__(self, p=0.5, distortion_scale=0.5):
        self.p = p
        self.distortion_scale = distortion_scale

    def __call__(self, img):
        if random.random() > self.p:
            return img

        width, height = img.size

        dx = int(self.distortion_scale * width)
        dy = int(self.distortion_scale * height)

        startpoints = [
            [0, 0],
            [width - 1, 0],
            [width - 1, height - 1],
            [0, height - 1]
        ]

        endpoints = [
            [random.randint(0, dx), random.randint(0, dy)],
            [random.randint(width - dx - 1, width - 1), random.randint(0, dy)],
            [random.randint(width - dx - 1, width - 1), random.randint(height - dy - 1, height - 1)],
            [random.randint(0, dx), random.randint(height - dy - 1, height - 1)]
        ]

        # Перспективное преобразование
        return F.perspective(img, startpoints, endpoints, interpolation=InterpolationMode.BILINEAR)


class RandomBrightness:
    def __init__(self, brightness=(0.5, 1.5), p=0.5):
        self.brightness = brightness
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        img = to_tensor(img)
        brightness_factor = random.uniform(*self.brightness)

        img = img * brightness_factor
        img = to_pil_image(img)
        return img
