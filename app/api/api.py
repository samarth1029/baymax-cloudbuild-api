import torchvision
import torchxrayvision as xrv
import torch

from utils.read_image import read_as_skimg
from base.gpt2lm import load_chatbot_model,generate_response,get_model_path


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

    @staticmethod
    def generate(input_text:str):
        model_path = get_model_path()
        model, tokenizer = load_chatbot_model(model_path)
        response = generate_response(model, tokenizer, input_text)
        print("AI Assistant:", response)
        return response
