import pyfeats
import numpy as np 

def feature_extraction(image):
    dic = {}
    mask = np.full((300, 300), 1)
    # For each channel of the image
    for i in range(image.shape[-1]):
        channel = image[:,:,i]
        features, labels = pyfeats.fos(channel, mask)
        features_mean, features_range, labels_mean, labels_range = pyfeats.glcm_features(channel, ignore_zeros=True)
        feats = np.concatenate((features, features_mean, features_range))
        dic[i] = feats
    feat_labels = labels + labels_mean + labels_range
    features = np.empty((3,44))
    for i in dic:
        features[i] = dic[i]
    return features, feat_labels