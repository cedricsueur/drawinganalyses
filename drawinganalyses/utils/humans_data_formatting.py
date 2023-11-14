import pandas as pd
import os
import fnmatch
import keras_tuner as kt


from drawinganalyses.config import LOCAL_DATA_DIR
data_dir = os.path.join(LOCAL_DATA_DIR, 'Humans')
labels = {'Crèche': 0, 'PS': 0, 'MS': 0, 'GS': 0, 'CE1': 0, 'CE2': 0, 'CM1': 0, 'CM2': 0, 'NOV': 1, 'EXP': 1}

drawing_filenames = fnmatch.filter(os.listdir(str(data_dir)), '*.jpg')
list_tuple_annotations = []

for filename in os.listdir(data_dir):
    folder = os.path.join(data_dir, filename)
    # checking if it is a file
    if not os.path.isfile(folder):
        for file in os.listdir(folder):
            ext = os.path.splitext(file)[-1].lower()
            if ext == ".jpg":
                label_folder = os.path.basename(folder)
                list_tuple_annotations.append((label_folder+'/'+file, labels[label_folder]))

df = pd.DataFrame(list_tuple_annotations)
df.columns = ['name', 'label']
df.to_csv(os.path.join(data_dir, 'labels.csv'), index=False)