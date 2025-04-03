import os
import torch
from PIL import Image
from typing import Callable, Optional, Tuple, List

class WilddashDataset2(object):
    """
    Wilddash dataset loader.
    Assumes the following folder structure:
    wilddash/
            /images/*.jpg
            /labels/*.png
            /random_split/{split}.txt
    """
    def __init__(
        self,
        root: str,
        split: str,
        target_type: str = 'semantic',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        super(WilddashDataset2, self).__init__()

        self.split = split
        self.target_type = target_type
        self.transform = transform
        self.target_transform = target_transform
        self.transforms = transforms

        images_root = os.path.join(root, "images")
        targets_root = os.path.join(root, "labels")
        split_file = os.path.join(root, f"random_split/{split}.txt")
        
        if not os.path.exists(split_file):
            raise FileNotFoundError(f"Split file {split_file} not found!")
        
        with open(split_file, "r") as f:
            image_ids = [line.strip() for line in f.readlines()]

        self.images = [os.path.join(images_root, f"{img_id}.jpg") for img_id in image_ids]
        
        # Load the target labels based on the target_type
        if self.target_type == 'semantic':
            self.targets = [os.path.join(targets_root, f"{img_id}_labelIds.png") for img_id in image_ids]
        # Load the original raw segmentation labels
        else:
            self.targets = [os.path.join(targets_root, f"{img_id}.png") for img_id in image_ids]

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.targets[index])

        # if self.target_type == 'semantic':
        #     target = torch.tensor(target, dtype=torch.long)

        if self.transforms is not None:
            image, target = self.transforms(image, target)
            return image, target
        if self.transform is not None:
            image = self.transform(image)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return image, target

    def __len__(self) -> int:
        return len(self.images)


def wilddash2(root: str,
             split: str,
             transforms: List[Callable],
             target_type: str = 'semantic'):
    return WilddashDataset2(root=root,
                           split=split,
                           transforms=transforms,
                           target_type=target_type)
