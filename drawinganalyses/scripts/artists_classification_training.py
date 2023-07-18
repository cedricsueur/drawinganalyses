from drawinganalyses.datasets.drawings_pytorch import DrawingModule
from drawinganalyses.config import LOCAL_DATA_DIR
from torch.utils.data import random_split
from drawinganalyses.models.pytorch_models import TransferLearningModel
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

wandb_logger = WandbLogger()
trainer = Trainer(logger=wandb_logger)


def main():
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

if __name__ == '__main__':
    main()