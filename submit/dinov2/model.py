import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_classes = 19):
        super(Model, self).__init__()
        self.vit = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14', pretrained=True)  
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(384, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, num_classes, kernel_size=2, stride=2),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        feats = self.vit.forward_features(x)['x_prenorm'][:, 1:, :]
        b, n, c = feats.shape
        h = w = int(n ** 0.5)
        feats = feats.reshape(b, h, w, c).permute(0, 3, 1, 2)
        output = self.decoder(feats)
        return output

