import os
from pathlib import Path

LOCAL_DATA_DIR = Path(os.environ.get("MADE_DATA_DIR"))
assert LOCAL_DATA_DIR.exists()
MODELS_STORAGE = LOCAL_DATA_DIR / "models"
assert MODELS_STORAGE.exists()

# Example for the human drawings dataset
# DATASET_NAME = "Humans"
# ANNOTATION_FILE = "labels.csv"
# MODEL_NAME = f"{DATASET_NAME}_model.pth"
# label_to_str = {0: 'Child', 1: 'Adult'}
# INTERPRETABILITY_STORAGE = LOCAL_DATA_DIR / f"{DATASET_NAME}_interpretability"
# assert INTERPRETABILITY_STORAGE.exists()

# Example for the Molly's drawings dataset
DATASET_NAME = "Molly"
ANNOTATION_FILE = "labels.csv"
MODEL_NAME = f"{DATASET_NAME}_model.pth"
label_to_str = {0: 'Autumn', 1: 'Spring', 2: 'Summer', 3: 'Winter'}
INTERPRETABILITY_STORAGE = LOCAL_DATA_DIR / f"{DATASET_NAME}_interpretability"
assert INTERPRETABILITY_STORAGE.exists()

# Example for artists classifications
# DATASET_NAME = 'Artistanimals'
# ANNOTATION_FILE = "labels.csv" 
# MODEL_NAME = f"{DATASET_NAME}_model.pth"
# label_to_str = {0: 'Molly', 1: 'Pigcasso', 2: 'Pocketswarhol'}
# INTERPRETABILITY_STORAGE = LOCAL_DATA_DIR / f"{DATASET_NAME}_interpretability"
# assert INTERPRETABILITY_STORAGE.exists()
