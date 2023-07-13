import os
import glob 

from drawinganalyses.datasets.drawing import Drawing
from tqdm import tqdm

class DrawingCollectionSplit():
    drawings: list[Drawing]
    def __init__(self, data_dir):
        drawings = []        
        seasons = ['Autumn','Winter','Spring','Summer']
        drawings_names = []
        for season in seasons:
            directory = data_dir + season
            names = (file for file in os.listdir(directory) if os.path.isfile(os.path.join(directory, file)))
            names = list(names)
            if drawings_names:
                drawings_names = drawings_names + names
            else:
                drawings_names = names

        for drawing_name in tqdm(drawings_names):
            if not drawing_name.startswith('.'):
                for season in seasons:
                    if season in drawing_name:
                        drawing = Drawing(drawing_name=drawing_name, label=season, data_dir=data_dir)
                        drawings.append(drawing)
        self.drawings = drawings
        
        
    def __getitem__(self, index):
        return self.drawings[index]

    def get_images(self):
        list_images = []
        for drawing in self.drawings:
            list_images.append(drawing.image)
        return list_images
            
    def get_labels(self):
        list_labels = []
        for drawing in self.drawings:
            list_labels.append(drawing.label)
        return list_labels
    
    def get_label_to_class_name(self):
        return self.drawings[0].label_to_str


import os
import glob 

from drawinganalyses.datasets.drawing import Drawing
from tqdm import tqdm

class DrawingCollection():
    drawings: list[Drawing]
    def __init__(self, data_dir):
        drawings = []        
        seasons = ['Autumn','Winter','Spring','Summer']
        drawings_names = []
        drawings_names = (file for file in os.listdir(data_dir) if os.path.isfile(os.path.join(data_dir, file)))

        for drawing_name in tqdm(drawings_names):
            if not drawing_name.startswith('.'):
                for season in seasons:
                    if season in drawing_name:
                        drawing = Drawing(drawing_name=drawing_name, label=season, data_dir=data_dir)
                        drawings.append(drawing)
        self.drawings = drawings
        
        
    def __getitem__(self, index):
        return self.drawings[index]

    def get_images(self):
        list_images = []
        for drawing in self.drawings:
            list_images.append(drawing.image)
        return list_images
            
    def get_labels(self):
        list_labels = []
        for drawing in self.drawings:
            list_labels.append(drawing.label)
        return list_labels
    
    def get_label_to_class_name(self):
        return self.drawings[0].label_to_str
