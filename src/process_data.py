import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as F

from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    Resize,
    ToDtype,
    Normalize,
)

def preprocess(img):
    """preproces image:
    input is a PIL image.
    Output image should be pytorch tensor that is compatible with your model"""
    mean = [0.485, 0.456, 0.406] # from ImageNet dataset
    std = [0.229, 0.224, 0.225] # from ImageNet dataset
    img_size = 644
    transform = Compose([
        ToImage(),
        Resize((img_size, img_size)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean, std),
    ])
    img = transform(img)
    img = img.unsqueeze(0)
    img = img.to(torch.float32) / 255.0
    return img

def postprocess(prediction, shape):
    """Post process prediction to mask:
    Input is the prediction tensor provided by your model, the original image size.
    Output should be numpy array with size [x,y,n], where x,y are the original size of the image and n is the class label per pixel.
    We expect n to return the training id as class labels. training id 255 will be ignored during evaluation."""
    m = torch.nn.Softmax(dim=1)
    prediction_soft = m(prediction)
    prediction_max = torch.argmax(prediction_soft, axis=1)
    prediction = transforms.functional.resize(prediction_max, size=shape, interpolation=transforms.InterpolationMode.NEAREST)

    prediction_numpy = prediction.cpu().detach().numpy()
    prediction_numpy = prediction_numpy.squeeze()

    return prediction_numpy
