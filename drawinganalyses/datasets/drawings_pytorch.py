import os
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
from PIL import Image
from torch.utils.data import random_split
from torch.utils.data import DataLoader
from torchvision import models, transforms
import pytorch_lightning as pl


class DrawingDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(img_dir +"/"+ annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        self.to_str_label = {0: 'Autumn', 1: 'Spring', 2: 'Summer', 3: 'Winter'}

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = Image.open(img_path).convert("RGB")
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def __str__(self):
        return ("Drawing name : {}, label : {}".format(self.drawing_name, self.to_str_label[self.label]))

    def show(self, idx):
        basewidth = 300
        img = Image.open("{data_dir}/{drawing_name}".format(data_dir=self.img_dir, drawing_name=self.img_labels.iloc[idx, 0]))
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), Image.Resampling.LANCZOS)

        display(img)
        return("Drawing name : {}, label : {}".format(self.img_labels.iloc[idx, 0], self.to_str_label[self.img_labels.iloc[idx, 1]]))


class DrawingModule(pl.LightningDataModule):
    def __init__(self, data_dir: str, annotation_file:str, batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.annotation_file = annotation_file
        self.transforms = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize the image to 256x256 pixels
            transforms.RandomHorizontalFlip(),  # Randomly flip the image horizontally
            transforms.ToTensor(),  # Convert the image to a PyTorch tensor
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize the image
        ])
    def setup(self, stage: str):
        # Assign train/val datasets for use in dataloaders
        if stage == "fit":
            draws = DrawingDataset(self.annotation_file, self.data_dir, transform=self.transforms)
            self.dataset_train, self.dataset_val = random_split(draws, [0.8, 0.2])
                # Assign test dataset for use in dataloader(s)
        if stage == "test":
            self.draw_test = DrawingDataset(self.annotation_file, self.data_dir, transform=self.transforms)

        if stage == "predict":
            self.draw_predict = DrawingDataset(self.annotation_file, self.data_dir, transform=self.tranforms)

    def train_dataloader(self):
        return DataLoader(self.dataset_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.dataset_val, batch_size=self.batch_size)
    
    def test_dataloader(self):
        return DataLoader(self.draw_test, batch_size=32)

    def predict_dataloader(self):
        return DataLoader(self.draw_predict, batch_size=32)