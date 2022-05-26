
from timm.data.transforms_factory import create_transform
from PIL import Image
import torch

class EfficientNetImagePreprocessor:
    
    def __init__(self, config):
        self.image_preprocessor = create_transform(**config)
 
    def __call__(self, img, *args, **kwargs):
        img_list = self._np_to_PIL(img)
        preprocessed_img_list = [self.image_preprocessor(img) for img in img_list]
        return {"images": torch.stack(preprocessed_img_list)}

    def _np_to_PIL(self, img):
        if isinstance(img, list):
            img = [Image.fromarray(x) for x in img]
        else:
            img = [Image.fromarray(img)]
        return img
    
    @classmethod
    def from_config(cls, config):
        return cls(config)