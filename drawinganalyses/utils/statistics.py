import torch.nn.functional as F
import torch
from captum.attr import IntegratedGradients
from captum.attr import GradientShap
from captum.attr import Occlusion
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz
import matplotlib.pyplot as plt
import numpy as np
import pickle

from drawinganalyses.config import LOCAL_DATA_DIR

def explinability_count(dataloader, model, label_to_str, class_names, threshold):
    season_count_0 = {'Winter':[], 'Summer':[], 'Autumn':[], 'Spring':[]}
    season_count_05 = {'Winter':[], 'Summer':[], 'Autumn':[], 'Spring':[]}
    season_count_1 = {'Winter':[], 'Summer':[], 'Autumn':[], 'Spring':[]}

    i=0
    a_file = open("data.pkl", "wb")
    for inputs, labels in iter(dataloader):
        
        true_label = label_to_str[labels.item()]
        output = model(inputs)
        output = F.softmax(output, dim=1)
        prediction_score, pred_label_idx = torch.topk(output, 1)
        pred_label_idx.squeeze_()
        predicted_label = class_names[(pred_label_idx.item())]
    
        torch.manual_seed(0)
        np.random.seed(0)
            
        occlusion = Occlusion(model)
                
        attributions_occ = occlusion.attribute(inputs,
                            strides = (3, 8, 8),
                            target=pred_label_idx,
                            sliding_window_shapes=(3,15, 15),
                            baselines=0)
        
        # Use boolean indexing to find positive elements
        positive_elements_0 = np.array(attributions_occ) > 0
        positive_elements_05 = np.array(attributions_occ) > 0.5
        positive_elements_1= np.array(attributions_occ) > 1


        # Count the number of positive elements
        count_positive_0 = np.sum(positive_elements_0)
        count_positive_05 = np.sum(positive_elements_05)
        count_positive_1 = np.sum(positive_elements_1)

        # Print the result
        print("i:", i)
        print("Season:", predicted_label)
        print("Number of positive elements:", count_positive_0)
        print("Number of positive elements > 0.5:", count_positive_05)
        print("Number of positive elements > 1:", count_positive_1)

        season_count_0[predicted_label].append(count_positive_0)
        season_count_05[predicted_label].append(count_positive_05)
        season_count_1[predicted_label].append(count_positive_1)

        i+=1
        file0 = open("data0.pkl", "wb")
        file05 = open("data05.pkl", "wb")
        file1 = open("data1.pkl", "wb")

        pickle.dump(season_count_0, file0)
        pickle.dump(season_count_05, file05)
        pickle.dump(season_count_1, file1)

        file0.close()
        file05.close()
        file1.close()

        if i == 400:
            break