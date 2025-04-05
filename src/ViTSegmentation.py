import torch
import torch.nn as nn
import torch.nn.init as init
from torchvision.transforms import functional as F

class ViTSegmentation(nn.Module):
    def __init__(self, num_classes):
        super(ViTSegmentation, self).__init__()
        
        self.convd = nn.Conv2d(4, 3, kernel_size=1, stride=1, padding=0)
        self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', 
                                  pretrained=True, force_reload=True)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(384, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Upsample(size=(112, 112), mode='bilinear', align_corners=True),
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        # Convd layer initialization
        init.xavier_uniform_(self.convd.weight)
        init.zeros_(self.convd.bias)
            
    def forward(self, x):
        # print(f"Shape input: {x.shape}")
        x = self.convd(x)
        feats = self.vit.forward_features(x)['x_prenorm'][:, 1:, :]
        b, n, c = feats.shape
        h = w = int(n ** 0.5)
        feats = feats.reshape(b, h, w, c).permute(0, 3, 1, 2)
        output = self.decoder(feats)
        return output

