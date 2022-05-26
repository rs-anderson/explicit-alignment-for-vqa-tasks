
from preprocessing import EfficientNetImagePreprocessor
import numpy as np
from PIL import Image
import torch


def test_image_preprocessor():
    batch_size = 2
    
    ImagePreprocessorConfig = {
        'input_size': [3, 224, 224],
        'interpolation': 'bicubic', 
        'mean': [0.485, 0.456, 0.406],
        'std': [0.229, 0.224, 0.225], 
        'crop_pct': 0.875
    }
    image = Image.new('RGB', (400, 600))
    images = [np.array(image) for _ in range(batch_size)]
    
    image_preproccesor = EfficientNetImagePreprocessor.from_config(ImagePreprocessorConfig)
    preprocessed_images = image_preproccesor(images)['images']

    expected_shape = (batch_size, 3, 224, 224)
    assert preprocessed_images.shape == expected_shape and isinstance(preprocessed_images, torch.Tensor)
