import torchvision
import torchxrayvision as xrv
import torch

from app import (
    __version__,
    __appname__,
    __email__,
    __author__,
)
from app.utils.read_image import read_as_skimg


class Api:
    @staticmethod
    def get_app_details() -> dict:
        return {
            "appname": __appname__,
            "version": __version__,
            "email": __email__,
            "author": __author__,
        }

    @staticmethod
    def get_xray_reports(url: str) -> dict:
        response = read_as_skimg(url)
        img = xrv.datasets.normalize(response, 255)
        img = img.mean(2)[None, ...]
        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(), xrv.datasets.XRayResizer(224)])
        img = transform(img)
        img = torch.from_numpy(img)
        model = xrv.models.DenseNet(weights="densenet121-res224-all")
        outputs = model(img[None, ...])
        print(dict(zip(model.pathologies, outputs[0].detach().numpy())))
        predictions = {
            pathology: prediction.item() for pathology, prediction in zip(model.pathologies, outputs[0].cpu())
        }
        return {"preds": predictions}
