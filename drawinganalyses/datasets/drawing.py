from PIL import Image
from dataclasses import dataclass

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
        
        basewidth = 300
        img = Image.open("{data_dir}/{drawing_name}".format(data_dir=self.data_dir, label=self.label_to_str[self.label], drawing_name=self.drawing_name))
        wpercent = (basewidth/float(img.size[0]))
        hsize = int((float(img.size[1])*float(wpercent)))
        img = img.resize((basewidth,hsize), Image.Resampling.LANCZOS)

        self.image = img
        
    def __str__(self):
        return ("Drawing name : {}, label : {}".format(self.drawing_name, self.label_to_str[self.label]))
        
    def __getitem__(self):
        print("Drawing name : {}, test label : {}".format(self.drawing_name, self.label_to_str[self.label]))
        return(self.image)

    def __repr__(self):
        display(self.image)
        return("Drawing name : {}, label : {}".format(self.drawing_name, self.label_to_str[self.label]))