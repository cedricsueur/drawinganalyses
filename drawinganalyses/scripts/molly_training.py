from drawinganalyses.datasets.drawings_pytorch import DrawingModule
from drawinganalyses.config import LOCAL_DATA_DIR
from torch.utils.data import random_split
from drawinganalyses.models.pytorch_models import TransferLearningModel
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint


def main():
    annotation_file = "labels.csv" 
    models_storage = LOCAL_DATA_DIR / "models"
    draws = DrawingModule(data_dir=LOCAL_DATA_DIR, annotation_file=annotation_file)

    model = TransferLearningModel()
    # Créez un objet ModelCheckpoint
    checkpoint_callback = ModelCheckpoint(dirpath=models_storage, filename='modele-{epoch:02d}-{val_loss:.2f}')

    # Créez votre trainer avec le callback
    trainer = Trainer(max_epochs=20, default_root_dir=models_storage)
    trainer.fit(model=model, datamodule=draws)

if __name__ == '__main__':
    main()