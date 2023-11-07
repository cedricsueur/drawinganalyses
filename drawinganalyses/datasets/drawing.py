from PIL import Image
from dataclasses import dataclass
from numpy import asarray

@dataclass
class Drawing():
    image: Image
    label: str
    set: str = None
    
    def __init__(self, drawing_name, label, data_dir):
        self.data_dir = data_dir
        self.drawing_name = drawing_name
        self.dict_label = {'Autumn': 0, 'Spring': 1, 'Summer': 2, 'Winter': 3}
        self.label_to_str = {0: 'Autumn', 1: 'Spring', 2: 'Summer', 3: 'Winter'}
        self.label = self.dict_label[label]
        self.path = "{data_dir}/{drawing_name}".format(data_dir=self.data_dir, label=self.label_to_str[self.label], drawing_name=self.drawing_name)
        img = Image.open("{data_dir}/{drawing_name}".format(data_dir=self.data_dir, label=self.label_to_str[self.label], drawing_name=self.drawing_name))
        #basewidth = 300
        #wpercent = (basewidth/float(img.size[0]))
        #hsize = int((float(img.size[1])*float(wpercent)))
        # img = img.resize((basewidth,hsize), Image.Resampling.LANCZOS)
        img = img.resize((300,300))

        self.image = asarray(img)
        
                
    def __str__(self):
        return ("Drawing name : {}, label : {}".format(self.drawing_name, self.label_to_str[self.label]))
        
    def __getitem__(self):
        return(self.image)

    def show(self):
        img = Image.fromarray(self.image)
        basewidth = 300
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), Image.Resampling.LANCZOS)
        return(img)