from drawinganalyses.datasets.drawings_pytorch import DrawingModule, DrawingDataset
from drawinganalyses.config import LOCAL_DATA_DIR
from torch.utils.data import random_split
from drawinganalyses.models.pytorch_models import TransferLearningModel, Net
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import torch

#wandb_logger = WandbLogger()
#trainer = Trainer(logger=wandb_logger)


def main_old():
    annotations_file = "labels.csv" 
    dataset_name = 'Artistanimals'
    models_storage = LOCAL_DATA_DIR / "models" / dataset_name
    label_to_str = {0: 'Molly', 1: 'Pigcasso', 2: 'Pocketswarhol'}
    
    
    draws = DrawingModule(
        dataset_name=dataset_name,
        data_dir=LOCAL_DATA_DIR,
        annotations_file=annotations_file,
        label_to_str=label_to_str)

    model = TransferLearningModel()
    
    # Créez votre trainer avec le callback
    
    wandb_logger = WandbLogger()
    trainer = Trainer(max_epochs=20, default_root_dir=models_storage, logger=wandb_logger)
    trainer.fit(model=model, datamodule=draws)
    
import torch.optim as optim
import torch.nn as nn
import torch
import torchvision
import torchvision.transforms as transforms

def main_naive():
    annotations_file = "labels.csv" 
    dataset_name = 'Artistanimals'
    models_storage = LOCAL_DATA_DIR / "models" / dataset_name
    label_to_str = {0: 'Molly', 1: 'Pigcasso', 2: 'Pocketswarhol'}
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
    batch_size = 8
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)
    net = Net()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f}')
                running_loss = 0.0

    print('Finished Training')
    PATH = './train_test.pth'
    torch.save(net.state_dict(), PATH)

from torchvision.models import resnet50

def main():
    
    annotations_file = "labels.csv" 
    dataset_name = 'Artistanimals'
    models_storage = LOCAL_DATA_DIR / "models" / dataset_name
    label_to_str = {0: 'Molly', 1: 'Pigcasso', 2: 'Pocketswarhol'}
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
    batch_size = 8
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

    # load up the ResNet50 model
    model = resnet50(pretrained=True)
    # since we are using the ResNet50 model as a feature extractor we set
    # its parameters to non-trainable (by default they are trainable)
    for param in model.parameters():
        param.requires_grad = False
    # append a new classification top to our feature extractor and pop it
    # on to the current device
    modelOutputFeats = model.fc.in_features
    model.fc = nn.Linear(modelOutputFeats, len(trainset.classes))

if __name__ == '__main__':
    main()