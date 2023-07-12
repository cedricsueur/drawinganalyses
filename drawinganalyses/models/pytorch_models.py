import logging
from pathlib import Path
from typing import Union
import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
import pytorch_lightning as pl
from pytorch_lightning import LightningDataModule
from pytorch_lightning.callbacks.finetuning import BaseFinetuning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping

class TransferLearningModel(pl.LightningModule):
    def __init__(
        self,
        backbone: str = "resnet50",
        train_bn: bool = False,
        milestones: tuple = (2, 4),
        batch_size: int = 32,
        lr: float = 1e-3,
        lr_scheduler_gamma: float = 1e-1,
        num_workers: int = 6,
        **kwargs,
    ) -> None:
        """TransferLearningModel.
        Args:
            backbone: Name (as in ``torchvision.models``) of the feature extractor
            train_bn: Whether the BatchNorm layers should be trainable
            milestones: List of two epochs milestones
            lr: Initial learning rate
            lr_scheduler_gamma: Factor by which the learning rate is reduced at each milestone
        """
        super().__init__()
        self.backbone = backbone
        self.train_bn = train_bn
        self.milestones = milestones
        self.batch_size = batch_size
        self.lr = lr
        self.lr_scheduler_gamma = lr_scheduler_gamma
        self.num_workers = num_workers
        self.__build_model()
        self.train_acc = Accuracy(task='multiclass', num_classes=4)
        self.valid_acc = Accuracy(task='multiclass', num_classes=4)
        self.save_hyperparameters()
        
    def __build_model(self):
        """Define model layers & loss."""
        # 1. Load pre-trained network:
        model_func = getattr(models, self.backbone)
        backbone = model_func(pretrained=True)
        _layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*_layers)# 2. Classifier:
        _fc_layers = [nn.Linear(2048, 256), nn.ReLU(), nn.Linear(256, 32), nn.Linear(32, 1)]
        self.fc = nn.Sequential(*_fc_layers)# 3. Loss:
        self.loss_func = F.binary_cross_entropy_with_logits
        
    def forward(self, x):
        """Forward pass.
        Returns logits.
        """
        # 1. Feature extraction:
        x = self.feature_extractor(x)
        x = x.squeeze(-1).squeeze(-1)
        # 2. Classifier (returns logits):
        x = self.fc(x)
        return x
        
    
    def loss(self, logits, labels):
            return self.loss_func(input=logits, target=labels)
        
    def training_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_scores = torch.sigmoid(y_logits)
        y_true = y.view((-1, 1)).type_as(x)
        # 2. Compute loss
        train_loss = self.loss(y_logits, y_true)
        # 3. Compute accuracy:
        self.log("train_acc", self.train_acc(y_scores, y_true.int()), prog_bar=True)
        return train_loss
    
    def validation_step(self, batch, batch_idx):
        # 1. Forward pass:
        x, y = batch
        y_logits = self.forward(x)
        y_scores = torch.sigmoid(y_logits)
        y_true = y.view((-1, 1)).type_as(x)
        # 2. Compute loss
        self.log("val_loss", self.loss(y_logits, y_true), prog_bar=True)
        # 3. Compute accuracy:
        self.log("val_acc", self.valid_acc(y_scores, y_true.int()), prog_bar=True)
        
    def test_step(self, batch, batch_idx):
        x, y = batch
        y_logits = self.forward(x)
        y_scores = torch.sigmoid(y_logits)
        y_true = y.view((-1, 1)).type_as(x)
        self.log("test_acc", self.valid_acc(y_scores, y_true.int()), prog_bar=True)

        
    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        optimizer = optim.Adam(trainable_parameters, lr=self.lr)
        scheduler = MultiStepLR(optimizer, milestones=self.milestones, gamma=self.lr_scheduler_gamma)
        return [optimizer], [scheduler]