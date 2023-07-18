import pandas as pd
import os, fnmatch
import shutil

from drawinganalyses.config import LOCAL_DATA_DIR

molly_data_dir = LOCAL_DATA_DIR / 'Molly'
Pigcasso_data_dir = LOCAL_DATA_DIR / 'Pigcasso'
Pocketswarhol_data_dir = LOCAL_DATA_DIR / 'Pocketswarhol'

data_dir = LOCAL_DATA_DIR / 'Artistanimals'

labels = {'Molly': 0, 'Pigcasso': 1, 'Pocketswarhol': 2}
list_tuple_annotations = []
list_dir = [molly_data_dir, Pigcasso_data_dir, Pocketswarhol_data_dir]
for directory in list_dir:
    for label in labels.keys():
        if label in str(directory):
            label_tmp=label
    for root, dirs, files in os.walk(directory):
        for drawing in files:
            if drawing.endswith(('.jpg', '.jpeg', '.png')):
                shutil.copy(directory / drawing, data_dir)
                list_tuple_annotations.append((drawing, labels[label_tmp]))
                
print(list_tuple_annotations[0])
df = pd.DataFrame(list_tuple_annotations)
df.columns = ['name', 'label']
df.to_csv(data_dir / 'labels.csv', index=False)