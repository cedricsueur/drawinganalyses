import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
from PIL import Image
from tempfile import TemporaryDirectory
from pathlib import Path
from drawinganalyses.datasets.drawings_pytorch import DrawingModule, DrawingDataset
from torch.utils.data import random_split
import torch
from drawinganalyses.utils.visualization import explinability_images, explinability_images_save

from drawinganalyses.config import LOCAL_DATA_DIR

cudnn.benchmark = True
plt.ion()   # interactive mode

def main():
    annotations_file = "labels.csv" 
    models_storage = LOCAL_DATA_DIR / "models"
    dataset_name = 'Molly'
    label_to_str = {0: 'Autumn', 1: 'Spring', 2: 'Summer', 3: 'Winter'}
    batch_size=4
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((256, 256))])

    generator1 = torch.Generator().manual_seed(42)
    dataset = DrawingDataset(
        dataset_name=dataset_name,
        annotations_file=annotations_file,
        data_dir=LOCAL_DATA_DIR,
        label_to_str=label_to_str,
        transform=transform)
    trainset, valset, testset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator1)

    class_names = list(label_to_str.values())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    PATH = './test_molly.pth'
    model_ft = models.resnet18(pretrained=True)
    num_ftrs = model_ft.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    model_ft.fc = nn.Linear(num_ftrs, len(class_names))

    model_ft = model_ft.to(device)
    model_ft.load_state_dict(torch.load(LOCAL_DATA_DIR/PATH))
    model_ft.eval()
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    
    explinability_images_save(trainloader, model_ft, label_to_str, class_names)
    
if __name__ == '__main__':
    main()