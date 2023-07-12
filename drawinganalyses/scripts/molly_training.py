from drawinganalyses.datasets.drawings_pytorch import DrawingModule
from drawinganalyses.config import LOCAL_DATA_DIR
from torch.utils.data import random_split
from drawinganalyses.models.pytorch_models import TransferLearningModel
from pytorch_lightning import Trainer

annotation_file = "labels.csv" 

models_storage = LOCAL_DATA_DIR / "models"
draws = DrawingModule(data_dir=LOCAL_DATA_DIR, annotation_file=annotation_file)

model = TransferLearningModel()
trainer = Trainer(max_epochs=20)
trainer.fit(model=model, datamodule=draws, default_root_dir=models_storage)