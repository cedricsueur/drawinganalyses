import pandas as pd
import os, fnmatch

from drawinganalyses.config import LOCAL_DATA_DIR

data_dir = LOCAL_DATA_DIR / 'Molly'
labels = {'Autumn': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}

drawing_filenames = fnmatch.filter(os.listdir(str(data_dir)), '*.jpg')
list_tuple_annotations = []
for drawing in drawing_filenames:
    for label in labels.keys():
        if label in drawing:
            list_tuple_annotations.append((drawing, labels[label]))
            
df = pd.DataFrame(list_tuple_annotations)
df.to_csv(data_dir / 'labels.csv', index=False)