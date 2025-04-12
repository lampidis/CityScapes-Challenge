import torch
import torch.nn as nn
import torch.nn.init as init
import urllib
# from torchvision.transforms import functional as F
import torch.nn.functional as F
from functools import partial
from torchinfo import summary


def resize(input_data,
       size=None,
       scale_factor=None,
       mode='nearest',
       align_corners=None,
       warning=True):
    if warning:
        if size is not None and align_corners:
            input_h, input_w = tuple(int(x) for x in input.shape[2:])
            output_h, output_w = tuple(int(x) for x in size)
    return F.interpolate(input_data, size, scale_factor, mode, align_corners)

def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()
    
class BNHead(nn.Module):
    """Just a batchnorm."""

    def __init__(self, num_classes=19, resize_factors=None, **kwargs):
        super().__init__(**kwargs)
        # HARDCODED IN_CHANNELS FOR NOW.
        self.in_channels = 1536 #*4 # sum([feature.shape[1] for feature in selected_features])
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.resize_factors = resize_factors
        self.in_index = [0, 1, 2, 3]
        self.input_transform = 'resize_concat'
        self.align_corners = False

        self.conv_seg = nn.Conv2d(self.in_channels, num_classes, kernel_size=1)

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

        if self.input_transform == "resize_concat":
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
            print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (len(self.resize_factors), len(inputs))
                inputs = [
                    resize(input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area")
                    for x, f in zip(inputs, self.resize_factors)
                ]
            print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(input_data=x, size=inputs[0].shape[2:], mode="bilinear", align_corners=self.align_corners)
                for x in inputs
            ]
            print("after upsampled", *(x.shape for x in upsampled_inputs))
            inputs = torch.cat(upsampled_inputs, dim=1)
            print("after cat", *(x.shape for x in inputs))
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def cls_seg(self, feat):
        """Classify each pixel."""
        output = self.conv_seg(feat)
        return output
        
    
    def forward(self, inputs):
        """Forward function."""
        print("bnhead forward", *(x.shape for x in inputs))
        output = self._forward_feature(inputs)
        print("bnhead _forward_feature", *(x.shape for x in inputs))
        output = self.cls_seg(output)
        print("bnhead output", *(x.shape for x in output))
        return output
    

class ViTSegmentation(nn.Module):
    def __init__(self, num_classes=19):
        super(ViTSegmentation, self).__init__()
        HEAD_DATASET = "voc2012" # in ("ade20k", "voc2012")
        HEAD_TYPE = "ms" # in ("ms, "linear")
        DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
        backbone_name = "dinov2_vits14"
        
        head_config_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_config.py"
        # head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{HEAD_DATASET}_{HEAD_TYPE}_head.pth"

        self.convd = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0)
        self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)
        
        cfg_str = load_config_from_url(head_config_url)
        
        # Define a namespace dict to get the config and then extract it
        namespace = {}
        exec(cfg_str, namespace)
        model_dict = namespace['model']
        print(f"self.vit.blocks : {len(self.vit.blocks)}")
        self.vit.forward = partial(
            self.vit.get_intermediate_layers,
            n=model_dict['backbone']['out_indices'],
            reshape=True,
        )
        
        self.decoder = BNHead(num_classes)
        self._initialize_weights()

    def _initialize_weights(self):
        # Convd layer initialization
        init.xavier_uniform_(self.convd.weight)
        init.zeros_(self.convd.bias)
            
    def forward(self, x):
        # print(f"Shape input: {x.shape}")
        x = self.convd(x)
        feats = self.vit(x)#.forward_features(x)['x_prenorm'][:, 1:, :]
        print(f"Shape encoder: {len(feats)}")
        # b, n, c = feats.shape
        # h = w = int(n ** 0.5)
        # feats = feats.reshape(b, h, w, c).permute(0, 3, 1, 2)
        output = self.decoder(feats)
        print(f"Shape decoder: {output.shape}")
        output = torch.nn.functional.interpolate(output, size=x.shape[2:], mode="bilinear", align_corners=False)
        print(f"Shape output: {output.shape}")
        return output

if __name__ == '__main__':
    model = ViTSegmentation()
    summary(
        model, 
        (1, 4, 644, 644),
        col_names=('input_size', 'output_size', 'num_params'),
        row_settings=['var_names']
    )