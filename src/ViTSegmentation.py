import torch
import torch.nn as nn
import torch.nn.init as init
import urllib

import numpy as np
import torch.nn.functional as F
from functools import partial
import mahalanobis as mh
# from saved_tensors import mean, cov

# from torchinfo import summary

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()
    

class AddFrequencyChannelTransform(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.0):
        super().__init__()
        self.kernel = self.gaussian_kernel(kernel_size, sigma)
        
    def gaussian_kernel(self, size: int, sigma: float):
        """Create a Gaussian kernel"""
        x = torch.linspace(-sigma, sigma, size)
        x = torch.exp(-x**2 / (2 * sigma**2))
        kernel = torch.outer(x, x)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)

    def __call__(self, batch_img):
        gray_batch = batch_img.mean(dim=1, keepdim=True)
        gray_convolved = nn.functional.conv2d(gray_batch, self.kernel, padding=self.kernel.size(2)//2)
        
        freq = gray_batch - gray_convolved
        
        return freq

class BNHead(nn.Module):
    """Just a batchnorm."""
    def __init__(self, resize_factors=None, num_classes=19):
        super().__init__()
        self.in_channels = 1536
        self.bn = nn.BatchNorm2d(self.in_channels)
        self.in_index = [0, 1, 2, 3]
        
        self.conv_seg = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)
        self.resize_factors = resize_factors

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        x = self._transform_inputs(inputs)
        feats = self.bn(x)
        return feats

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        # accept lists (for cls token)
        input_list = []
        for x in inputs:
            if isinstance(x, list):
                input_list.extend(x)
            else:
                input_list.append(x)
        inputs = input_list
        # an image descriptor can be a local descriptor with resolution 1x1
        for i, x in enumerate(inputs):
            if len(x.shape) == 2:
                inputs[i] = x[:, :, None, None]
        # select indices
        inputs = [inputs[i] for i in self.in_index]
        # Resizing shenanigans
        # print("before", *(x.shape for x in inputs))
        if self.resize_factors is not None:
            assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
            inputs = [
                F.interpolate(x, scale_factor=f, mode='bilinear' if f >= 1 else "area")
                # resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                for x, f in zip(inputs, self.resize_factors)
            ]
            # print("after", *(x.shape for x in inputs))
        upsampled_inputs = [
            F.interpolate(x, size=inputs[0].shape[2:], mode='bilinear', align_corners=False)
            # resize(input=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
            for x in inputs
        ]
        inputs = torch.cat(upsampled_inputs, dim=1)

        return inputs

    def cls_seg(self, feat):
        """Classify each pixel."""
        output = self.conv_seg(feat)
        return output
        
    
    def forward(self, inputs):
        """Forward function."""
        x = self._forward_feature(inputs)
        output = self.cls_seg(x)
        # x = self.upsample1(x)
        # output = self.upsample2(x)
        return output
    

class ViTSegmentation(nn.Module):
    def __init__(self, num_classes=19):
        super(ViTSegmentation, self).__init__()
        HEAD_DATASET = "voc2012" # in ("ade20k", "voc2012")
        HEAD_TYPE = "ms" # in ("ms, "linear")
        DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
        backbone_name = "dinov2_vits14"
        head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"

        self.vit = torch.hub.load('facebookresearch/dinov2', backbone_name, pretrained=True)
        scales =  [1.75, 2., 2., 2.] #[1.0, 1.32, 1.73]
        self.decoder = BNHead(resize_factors=scales, num_classes=19)
        
        for param in self.vit.parameters():
            param.requires_grad = False

        cfg_str = load_config_from_url(head_config_url)
        
        loaded = torch.load('mean_cov.pt')
        self.mean = loaded['mean']
        self.cov = loaded['cov']
        # self.mean = mean
        # self.cov = cov
        
        # namespace dict to get the config and then extract it
        namespace = {}
        exec(cfg_str, namespace)
        model_dict = namespace['model']
        self.vit.forward = partial(
            self.vit.get_intermediate_layers,
            n= model_dict['backbone']['out_indices'],
            reshape=True,
        )
        self.freq_conv = nn.Conv2d(num_classes+1, num_classes, kernel_size=3, padding=1)
        kernel = self.gaussian_kernel(kernel_size=5, sigma=1)  # Gaussian kernel
        self.gaussian_filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=5, padding=5 // 2, bias=False)
        self.gaussian_filter.weight.data = kernel
        self.gaussian_filter.weight.requires_grad = False

    
    def frequency_guided_predictions(self, logits, frequency_map, alpha=1.0):
        B, C, h, w = logits.shape
        
        # Step 2: Get predictions and one-hot masks
        soft_masks = F.softmax(logits, dim=1)  # [B, C, H, W]
        freq_weight = frequency_map.expand_as(soft_masks)  # [B, C, H, W]

        # Blend frequency into the soft masks
        guided_masks = soft_masks * (1.0 + alpha * freq_weight)

        # Optional: Normalize across class channel
        guided_masks = guided_masks / (guided_masks.sum(dim=1, keepdim=True) + 1e-8)

        return guided_masks

    def gaussian_kernel(self, kernel_size=5, sigma=1.0):
        """Create a Gaussian kernel"""
        x = torch.linspace(-sigma, sigma, kernel_size)
        x = torch.exp(-x**2 / (2 * sigma**2))
        kernel = torch.outer(x, x)
        kernel = kernel / kernel.sum()
        return kernel.unsqueeze(0).unsqueeze(0)  # Add batch and channel dims

    def frequency_response(self, batch_img):
        gray_batch = batch_img.mean(dim=1, keepdim=True)
        gray_convolved = self.gaussian_filter(gray_batch)
        freq = gray_batch - gray_convolved
        
        return freq
    
    def forward(self, x):
        B, C, h, w = x.shape
        feats = self.vit(x)
        
        mh_distances = []
        final_ood_score = 0
        print(f" feats shape: {feats[0].shape}")
        print(f" feats len: {len(feats)}")
        for b in range(B):
            # if itr==-1:
            distances = []
            for i in range(len(feats)):
                distances.append(mh.mahalanobis_distance(feats[i][b], self.mean[i], self.cov[i]))
            print(f"distances len: {len(distances)}")
            mh_distances.append(min(distances))
            # elif itr==0:
            #     self.mean[i], self.cov[i] = mh.batch_distribution(feats[i])
            # else:
            #     self.mean[i], self.cov[i] = mh.update_global_distribution(self.mean[i], self.cov[i], feats[i], itr)
            #     mean_cpu = [tensor for tensor in self.mean]
            #     cov_cpu = [tensor for tensor in self.cov]
            #     torch.save({'mean': mean_cpu, 'cov': cov_cpu}, 'mean_cov.pt')

        decoded = self.decoder(feats, )
        output = torch.nn.functional.interpolate(decoded, size=x.shape[2:], mode="bilinear", align_corners=False)
        
        freq_x = self.frequency_response(x)
        freq = torch.cat((output, freq_x), dim=1)
        output = self.freq_conv(freq)
        
        final_ood_score = mh_distances[0]
        print(f"mh_distances {len(mh_distances)}")
        print(f"final_ood_score {final_ood_score}")
        in_dist = False if final_ood_score > 19 else True
        return output, mh_distances

# if __name__ == '__main__':
#     model = ViTSegmentation()
#     summary(
#         model, 
#         (1, 4, 644, 644),
#         col_names=('input_size', 'output_size', 'num_params'),
#         row_settings=['var_names']
#     )