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
from drawinganalyses.utils.visualization import interpretability_save

from drawinganalyses.config import (
    LOCAL_DATA_DIR,
    DATASET_NAME,
    MODELS_STORAGE,
    ANNOTATION_FILE,
    MODEL_NAME,
    INTERPRETABILITY_STORAGE,
    label_to_str
)

import warnings
warnings.filterwarnings("ignore")

cudnn.benchmark = True
plt.ion()   # interactive mode

def main():
    transform = transforms.Compose(
        [transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        transforms.Resize((256, 256))])

    generator1 = torch.Generator().manual_seed(42)
    dataset = DrawingDataset(
        dataset_name=DATASET_NAME,
        annotations_file=ANNOTATION_FILE,
        data_dir=LOCAL_DATA_DIR,
        label_to_str=label_to_str,
        transform=transform)
    trainset, valset, testset = random_split(dataset, [0.8, 0.1, 0.1], generator=generator1)

    class_names = list(label_to_str.values())
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to ``nn.Linear(num_ftrs, len(class_names))``.
    model.fc = nn.Linear(num_ftrs, len(class_names))

    model = model.to(device)
    model.load_state_dict(torch.load(MODELS_STORAGE / MODEL_NAME))
    model.eval()
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=True, num_workers=2)
    valloader = torch.utils.data.DataLoader(valset, batch_size=1, shuffle=True, num_workers=2)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=2)

    print("Applying to the Training set")
    interpretability_save(trainloader, model, label_to_str, class_names, "train")
    print("Applying to the Validation set")
    interpretability_save(valloader, model, label_to_str, class_names, "valid")
    print("Applying to the Test set")
    interpretability_save(testloader, model, label_to_str, class_names, "test")

if __name__ == '__main__':
    main()