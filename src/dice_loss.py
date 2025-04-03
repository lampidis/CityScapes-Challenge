import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6, ignore_index=255):
        super(DiceLoss, self).__init__()
        self.smooth = smooth  # Avoid division by zero
        self.ignore_index = ignore_index

    def forward(self, logits, targets):
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=1)  # [batch_size, num_classes, H, W]
        
        # Get dimensions
        batch_size, num_classes = logits.shape[0], logits.shape[1]

        # Create a mask to exclude ignore_index
        if self.ignore_index is not None:
            mask = targets != self.ignore_index
        else:
            mask = torch.ones_like(targets, dtype=torch.bool)

        # Filter targets to exclude ignore_index before one-hot encoding
        valid_targets = targets.clone()
        valid_targets[~mask] = 0  # Set ignored values to a valid index (e.g., 0), but they’ll be masked out later

        # Convert to one-hot, only for valid class indices
        targets_one_hot = F.one_hot(
            valid_targets, num_classes=num_classes
        ).permute(0, 3, 1, 2).float()  # [batch_size, num_classes, H, W]

        # Apply mask to both probs and targets_one_hot
        probs = probs * mask.unsqueeze(1)  # Broadcast mask to channel dim
        targets_one_hot = targets_one_hot * mask.unsqueeze(1)

        # Flatten for Dice computation
        probs = probs.view(batch_size, num_classes, -1)  # [batch_size, num_classes, H*W]
        targets_one_hot = targets_one_hot.view(batch_size, num_classes, -1)

        # Compute intersection and union
        intersection = (probs * targets_one_hot).sum(dim=2)  # [batch_size, num_classes]
        union = probs.sum(dim=2) + targets_one_hot.sum(dim=2)  # [batch_size, num_classes]

        # Dice coefficient per class, then average across classes
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        dice = dice.mean(dim=1)  # Average over classes, [batch_size]
        
        # Return loss (1 - Dice)
        return 1 - dice.mean()