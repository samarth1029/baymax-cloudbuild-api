import torchxrayvision as xrv
import skimage, torch, torchvision


class ImagePreprocessor:
    def __init__(self, image):
        self.img = image

    def normalize_img(self):
        """
        convert 8-bit image to [-1024, 1024] range and return single color channel
        """
        self.img = xrv.datasets.normalize(self.img, 255)  # convert 8-bit image to [-1024, 1024] range
        return self.img.mean(2)[None, ...]

    def torchvision_transform(self):
        return torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
