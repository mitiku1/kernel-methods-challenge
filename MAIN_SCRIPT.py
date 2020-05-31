import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from skopt.space import Real
from skopt.utils import use_named_args
from skopt import gp_minimize
from sklearn.model_selection import cross_val_score
from collections import namedtuple
from utils.kernels import sigma_from_median
import os
from utils.classifiers import KernelSVMMultiK

import argparse



def transform_zero_one(y):
    return 2 * y - 1 
def transform_minus_one_one(y):
    return (y + 1)/2

def accuracy(y_pred, y_true):
    return np.mean((y_pred == y_true).astype(float))

def norm_normalization(inputs, test_inputs):
    norm = np.linalg.norm(inputs)

    return inputs/norm, test_inputs/norm
def std_normalization(inputs, test_inputs):
    mean = inputs.mean(axis=0)
    std = inputs.std(axis=0)

    inputs = (inputs - mean)/std
    test_inputs = (test_inputs - mean)/std
    return inputs, test_inputs
def min_max_normalization(inputs, test_inputs):
    min_ = inputs.min()
    max_ = inputs.max()
    inputs = (inputs - min_)/(min_  - max_)
    test_inputs = (test_inputs - min_)/(min_  - max_)
    return inputs, test_inputs
def threshold_normalize(inputs, test_inputs):
    mean = np.abs(inputs).mean()
    inputs[(-mean<inputs) & (inputs < mean)] = 0
    test_inputs[(-mean<test_inputs) & (test_inputs < mean)] = 0

    return min_max_normalization(inputs, test_inputs)
def get_features(files):
    output = []
    for f in files:
        output.append(pd.read_csv(f, header=None).values) 

    return output

def normalize_features(features,  test_features, normalize_function):
    for i in range(len(features)):
        features[i], test_features[i] = normalize_function(features[i], test_features[i])
    return features, test_features

itr = 0
def main():
    
    deg = 2
    kernel = "rbf"
   

    C = 100
    sigma = 0.009
    normalize = norm_normalization

    cases = [
        [3, 1],
        [4, 1],
        [5, 1], 
        [5, 2], 
        [5, 3], 
        [6, 1], 
        [6, 2], 
        [6, 3], 
        [7, 1]

    ]
    weights = np.array([
        2, 2, 3, 3, 3, 1, 1, 1, 1
    ])
    weights = weights/weights.sum()

    train_features_files = []
    for case in cases:
        train_features_files.append("logs/train_features-k-{}-m-{}.csv".format(*case))
    test_features_files = []
    for case in cases:
        test_features_files.append("logs/test_features-k-{}-m-{}.csv".format(*case))

    
    labels_path = "dataset/Ytr.csv"

    train_features = get_features(train_features_files)
    test_features = get_features(test_features_files)
    
    


    labels = pd.read_csv(labels_path)["Bound"].values.astype(np.double)

    labels = transform_zero_one(labels)

    train_features, test_features = normalize_features(train_features, test_features, normalize)
   

    


    clf = KernelSVMMultiK(kernel=kernel, sigma = sigma, C = C, degree=deg, weights=weights)

    
    

    *train_val_features,  train_labels, val_labels = \
                train_test_split(*train_features, labels, test_size=0.2)
    
    train_inputs = []
    val_inputs = []
    for i in range(len(train_val_features)):
        if i%22==0:
            train_inputs.append(train_val_features[i])
        else:
            val_inputs.append(train_val_features[i])

    




    
    
    clf.fit(train_inputs, train_labels)

    training_acc = accuracy(clf.predict(train_inputs), train_labels)
    valid_acc = accuracy(clf.predict(val_inputs), val_labels)
    print("trainin acc", training_acc)
    print("valid acc", valid_acc)
    
   
    test_pred = clf.predict(test_features)
    test_pred = transform_minus_one_one(test_pred).astype(np.int)
    
    test_df = pd.DataFrame(dict(Id=np.arange(test_pred.shape[0]),  Bound=test_pred))
    
    test_df.to_csv("submission.csv", index=False)





if __name__ == '__main__':
    
    main()

    