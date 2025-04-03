import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.transforms import functional as F

from PIL import Image
import scipy.ndimage as ndi


def preprocess(img):
    """preproces image:
    input is a PIL image.
    Output image should be pytorch tensor that is compatible with your model"""
    img = F.resize(img, size=(224, 224), interpolation=transforms.functional.InterpolationMode.BILINEAR)
    trans = transforms.Compose([transforms.ToTensor()])
    img = trans(img)
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


class AddFrequencyChannelTransform:
    def __init__(self, kernel_size=5, sigma=1.0):
        self.kernel = self.gaussian_kernel(kernel_size, sigma)  # Gaussian kernel
        
    def gaussian_kernel(self, size: int, sigma: float):
        """Create a Gaussian kernel"""
        x = torch.linspace(-sigma, sigma, size)
        x = torch.exp(-x**2 / (2 * sigma**2))
        kernel = torch.outer(x, x)
        kernel = kernel / kernel.sum()  # Normalize the kernel
        return kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

    def __call__(self, batch_img):
        # Convert to grayscale (mean across RGB channels)
        gray_batch = batch_img.mean(dim=1, keepdim=True)
        
        # Apply Gaussian convolution to the grayscale images
        gray_convolved = nn.functional.conv2d(gray_batch, self.kernel, padding=self.kernel.size(2)//2)
        
        freq = gray_batch - gray_convolved
        # Concatenate the grayscale image (after convolution) to the original RGB image
        concatenated_batch = torch.cat((batch_img, freq), dim=1)
        
        return concatenated_batch








