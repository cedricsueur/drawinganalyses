import pandas as pd
import os
import fnmatch


from drawinganalyses.config import LOCAL_DATA_DIR
data_dir = os.path.join(LOCAL_DATA_DIR, 'Humans')
labels = {'Crèche': 0, 'PS': 1, 'MS': 2, 'GS': 3, 'CE1': 4, 'CE2': 5, 'CM1': 6, 'CM2': 7, 'NOV': 8, 'EXP': 9}

drawing_filenames = fnmatch.filter(os.listdir(str(data_dir)), '*.jpg')
list_tuple_annotations = []

for filename in os.listdir(data_dir):
    folder = os.path.join(data_dir, filename)
    if not os.path.isfile(folder):
        for file in os.listdir(folder):
            ext = os.path.splitext(file)[-1].lower()
            if ext == ".jpg":
                label_folder = os.path.basename(folder)
                list_tuple_annotations.append((label_folder+'/'+file, labels[label_folder]))

df = pd.DataFrame(list_tuple_annotations)
df.columns = ['name', 'label']
df.to_csv(os.path.join(data_dir, 'labels.csv'), index=False)

