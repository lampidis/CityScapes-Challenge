"""
This script implements a training loop for the model. It is designed to be flexible, 
allowing you to easily modify hyperparameters using a command-line argument parser.

### Key Features:
1. **Hyperparameter Tuning:** Adjust hyperparameters by parsing arguments from the `main.sh` script or directly 
   via the command line.
2. **Remote Execution Support:** Since this script runs on a server, training progress is not visible on the console. 
   To address this, we use the `wandb` library for logging and tracking progress and results.
3. **Encapsulation:** The training loop is encapsulated in a function, enabling it to be called from the main block. 
   This ensures proper execution when the script is run directly.

Feel free to customize the script as needed for your use case.
"""
import os
from argparse import ArgumentParser

import wandb
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import Cityscapes
from wilddashdataset import WilddashDataset2
from torchvision.utils import make_grid
from torchvision.transforms import functional as F
import numpy as np
from torchvision.transforms.v2 import (
    Compose,
    ToImage,
    Resize,
    ToDtype,
    Normalize,
    InterpolationMode,
)
from tqdm import tqdm
import matplotlib.pyplot as plt

# from unet import Model
from ViTSegmentation import ViTSegmentation
from dice_loss import DiceLoss
from collections import defaultdict
# import segmentation_models_pytorch as smp

# Mapping class IDs to train IDs
id_to_trainid = defaultdict(lambda: 255, {cls.id: cls.train_id for cls in Cityscapes.classes})
id_to_trainid[34] = 13
id_to_trainid[35] = 13
# id_to_trainid = {cls.id: cls.train_id for cls in Cityscapes.classes}
def convert_to_train_id(label_img: torch.Tensor) -> torch.Tensor:
    return label_img.apply_(lambda x: id_to_trainid[x])


def convert_train_id_to_color(prediction: torch.Tensor) -> torch.Tensor:
    train_id_to_color = {cls.train_id: cls.color for cls in Cityscapes.classes if cls.train_id != 255}
    train_id_to_color[255] = (0, 0, 0)  # Assign black to ignored labels
    batch, _, height, width = prediction.shape
    color_image = torch.zeros((batch, 3, height, width), dtype=torch.uint8)

    for train_id, color in train_id_to_color.items():
        mask = prediction == train_id
                
        for i in range(3):
            color_image[:, i][mask.squeeze(1)] = color[i]

    return color_image

def get_args_parser():

    parser = ArgumentParser("Training script for a PyTorch dinov2 model")
    parser.add_argument("--data-dir", type=str, default="./data/cityscapes", help="Path to the training data")
    parser.add_argument("--batch-size", type=int, default=64, help="Training batch size")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--num-workers", type=int, default=6, help="Number of workers for data loaders")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--experiment-id", type=str, default="dinov2", help="Experiment ID for Weights & Biases")
    parser.add_argument("--wandb-save", type=bool, default=True, help="Save wandb logs flag")

    return parser


def main(args):
    # Initialize wandb for logging
    wandb.init(
        project="5lsm0-cityscapes-segmentation-dinov2",  # Project name in wandb
        name=args.experiment_id,  # Experiment name in wandb
        config=vars(args),  # Save hyperparameters
    ) if args.wandb_save else None

    # Create output directory if it doesn't exist
    output_dir = os.path.join("checkpoints", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Set seed for reproducability
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    # Define the device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Define the transforms to apply to the data
    mean = [0.485, 0.456, 0.406] # from ImageNet dataset
    std = [0.229, 0.224, 0.225] # from ImageNet dataset
    img_size = 644
    transform = Compose([
        ToImage(),
        Resize((img_size, img_size)),
        ToDtype(torch.float32, scale=True),
        Normalize(mean, std),
    ])
    target_transform = Compose([
        ToImage(),
        Resize((img_size, img_size), interpolation=InterpolationMode.NEAREST),
    ])

    # Load the dataset and make a split for training and validation
    train_cityscapes_dataset = Cityscapes(
        "./data/cityscapes", 
        split="train", 
        mode="fine", 
        target_type="semantic", 
        transform=transform,
        target_transform=target_transform,
    )
    valid_cityscapes_dataset = Cityscapes(
        "./data/cityscapes", 
        split="val", 
        mode="fine", 
        target_type="semantic", 
        transform=transform, 
        target_transform=target_transform,
    )
    train_wilddash_dataset = WilddashDataset2(
        "./data/wilddash2", 
        split="train",
        target_type="semantic",
        transform=transform, 
        target_transform=target_transform,
    )
    valid_wilddash_dataset = WilddashDataset2(
        "./data/wilddash2", 
        split="val",
        target_type="semantic",
        transform=transform, 
        target_transform=target_transform,
    )
    
    train_dataset = ConcatDataset([train_cityscapes_dataset, train_wilddash_dataset])
    valid_dataset = ConcatDataset([valid_cityscapes_dataset, valid_wilddash_dataset])
    print(f"Total train dataset size: {len(train_dataset)}")
    print(f"Total valid dataset size: {len(valid_dataset)}")
    
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_dataloader = DataLoader(
        valid_dataset, 
        batch_size=args.batch_size, 
        shuffle=False,
        num_workers=args.num_workers
    )

    # Define the model
    model = ViTSegmentation(num_classes=19)
    model.load_state_dict(torch.load("./checkpoints/dinov2_upsample/best_model-epoch=0029-val_loss=0.25763117604785496.pth"))
    model.to(device)
    
    # Define the loss function
    criterion = DiceLoss(ignore_index=255)  # Ignore the void class
    
    # Define the optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr)
    
    # Training loop
    best_valid_loss = float('inf')
    current_best_model_path = None
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1:04}/{args.epochs:04}")
        
        # Training
        model.train()
        for i, (images, labels) in tqdm(enumerate(train_dataloader)):
            # if i>1: break
            
            labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
            images, labels = images.to(device), labels.to(device)
            
            labels = labels.long().squeeze(1)  # Remove channel dimension

            optimizer.zero_grad()
            outputs, distances = model(images)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            wandb.log({
                "train_loss": loss.item(),
                "learning_rate": optimizer.param_groups[0]['lr'],
                "epoch": epoch + 1,
            }, step=epoch * len(train_dataloader) + i) if args.wandb_save else None
            
        # Validation
        model.eval()
        with torch.no_grad():
            losses = []
            all_distances = []
            for i, (images, labels) in tqdm(enumerate(valid_dataloader)):
                if i>1: break
                labels = convert_to_train_id(labels)  # Convert class IDs to train IDs
                images, labels = images.to(device), labels.to(device)
                
                labels = labels.long().squeeze(1)  # Remove channel dimension
                
                outputs, distances = model(images)
                distances = [d.cpu().numpy() for d in distances]
                all_distances.extend(distances)
                
                loss = criterion(outputs, labels)
                losses.append(loss.item())
            
                if i == 0:
                    predictions = outputs.softmax(1).argmax(1)

                    predictions = predictions.unsqueeze(1)
                    labels = labels.unsqueeze(1)

                    predictions = convert_train_id_to_color(predictions)
                    labels = convert_train_id_to_color(labels)

                    predictions_img = make_grid(predictions.cpu(), nrow=8)
                    labels_img = make_grid(labels.cpu(), nrow=8)

                    predictions_img = predictions_img.permute(1, 2, 0).numpy()
                    labels_img = labels_img.permute(1, 2, 0).numpy()
                    
                    print(f"prediction : {predictions_img.shape}")
                    print(f"labels : {labels_img.shape}")
                            
                    wandb.log({
                        "predictions": [wandb.Image(predictions_img)],
                        "labels": [wandb.Image(labels_img)],
                    }, step=(epoch + 1) * len(train_dataloader) - 1) if args.wandb_save else None
            
            valid_loss = sum(losses) / len(losses)
            wandb.log({
                "valid_loss": valid_loss
            }, step=(epoch + 1) * len(train_dataloader) - 1) if args.wandb_save else None
            
            print(f"mean_loaded: min:{min(all_distances)} max:{max(all_distances)}, mean:{sum(all_distances) / len(all_distances)}")
            print(f"threshold 95 -> {np.percentile(all_distances, 95)}")
            print(f"threshold 98 -> {np.percentile(all_distances, 98)}")
            print(f"validation loss: {valid_loss}")
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                if current_best_model_path:
                    os.remove(current_best_model_path)
                current_best_model_path = os.path.join(
                    output_dir, 
                    f"best_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
                )
                torch.save(model.state_dict(), current_best_model_path)
    print("Training complete!")

    
    # Save the model
    torch.save(
        model.state_dict(),
        os.path.join(
            output_dir,
            f"final_model-epoch={epoch:04}-val_loss={valid_loss:04}.pth"
        )
    )
    wandb.finish() if args.wandb_save else None

if __name__ == "__main__":
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
