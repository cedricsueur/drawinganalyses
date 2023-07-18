# Install

```
pip install -r requirements.txt -e .
```

Add MADE_LOCAL_DATA environment variable pointing to where your data are stored :

```
export MADE_LOCAL_DATA = /path/to/data_folder
```

Expected folder organization:

```
Data_folder
├── Molly
│   ├── img1.jpg
│   ├── img2.jpg
│   ├── img3.jpg
│   └── labels.csv
└── Humans
    ├── img1.jpg
    ├── img2.jpg
    ├── img3.jpg
    └── labels.csv
```

Expected data format for labels.csv:

```
labels.csv
├── img1,0
├── img2,1
├── img3,4
├── img4,0
```

# Configuration

## Training 

An example of training is provided in `drawinganalyses/scripts/molly_training.py`. Please adapt the main to your dataset name.
