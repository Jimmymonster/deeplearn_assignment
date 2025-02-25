import torch
from torchvision import transforms
from torchvision import models
import torch.nn as nn
import os
from PIL import Image
from torchvision.transforms import functional as F

def pad_to_square(image):
    w, h = image.size
    max_side = max(w, h)
    padding = (
        (max_side - w) // 2,
        (max_side - h) // 2,
        (max_side - w + 1) // 2,
        (max_side - h + 1) // 2
    )
    return transforms.functional.pad(image, padding, fill=0, padding_mode='constant')

transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip horizontally with 50% probability
    transforms.RandomRotation(degrees= 15),  # Randomly rotate the image by up to 90 degrees
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05, hue=0.05),  # Randomly change brightness, contrast, saturation, and hue
    # transforms.RandomPerspective(distortion_scale=0.1, p=0.5),
    transforms.Lambda(pad_to_square),
    # transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),  # for resnet
    transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),    # for inception
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Use RGB normalization
])

transform_val = transforms.Compose([
    transforms.Lambda(pad_to_square),
    # transforms.Resize(224, interpolation=transforms.InterpolationMode.BILINEAR),  # for resnet
    transforms.Resize(299, interpolation=transforms.InterpolationMode.BILINEAR),    # for inception
    transforms.ToTensor(),  # Converts to [0, 1] range
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
# model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
# model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

dataset_base_path = "data/banana/banana_splited_augmented"
dataset_train_path = os.path.join(dataset_base_path,"train")
dataset_validate_path = os.path.join(dataset_base_path,"eval")
dataset_test_path = os.path.join(dataset_base_path,"test")

class_num = len(os.listdir(dataset_train_path))

#resnet50
# model.fc = nn.Sequential(
#     nn.Linear(model.fc.in_features, 1024),
#     nn.BatchNorm1d(1024),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(1024, 512),
#     nn.BatchNorm1d(512),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(512, class_num),
# )

#resnet18
# model.fc = nn.Sequential(
#     nn.Linear(model.fc.in_features, 256),
#     nn.BatchNorm1d(256),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(256, 128),
#     nn.BatchNorm1d(128),
#     nn.ReLU(),
#     nn.Dropout(0.5),
#     nn.Linear(128, class_num),
# )

#inception
model.fc = nn.Sequential(
    nn.Linear(model.fc.in_features, 1024),
    nn.BatchNorm1d(1024),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(1024, 512),
    nn.BatchNorm1d(512),
    nn.ReLU(),
    nn.Dropout(0.5),
    nn.Linear(512, class_num),
)
