
from timm.data.transforms_factory import create_transform


class EfficientNetImagePreprocessor:
    
    def __init__(self, config):
        self.image_preprocessor = create_transform(**config)
 
    def __call__(self, img):
        return self.image_preprocessor(img)