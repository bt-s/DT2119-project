#!/usr/bin/python3

"""split_dataset.py: Split the data in usable training, validation and test sets

Usage from CLI: $ python3 train_val_test.py <fname>
Where <fname> specifies the name of the output .npz file containing all the data

Part of a project for the 2019 DT2119 Speech and Speaker Recognition course at
KTH Royal Institute of Technology"""

__author__ = "Pietro Alovisi, Romain Deffayet & Bas Straathof"


import pickle 
import numpy as np

from sklearn.model_selection import train_test_split
from collections import OrderedDict


if __name__ == "__main__":
    # Load all training data
    train_data = np.load("preprocessed_data/IRMAS-TrainingData_features.npy", 
            allow_pickle=True)

    # Load all testing data
    test_data1 = np.load("preprocessed_data/IRMAS-TestingData-Part1_features.npy", 
            allow_pickle=True)
    test_data2 = np.load("preprocessed_data/IRMAS-TestingData-Part2_features.npy", 
            allow_pickle=True)
    test_data3 = np.load("preprocessed_data/IRMAS-TestingData-Part3_features.npy", 
            allow_pickle=True)
    test_data = np.concatenate([test_data1, test_data2, test_data3])

    # Initialize the training feature and label matrices
    X_train = np.zeros((20115, 128, 43))
    y_train = np.zeros((20115, 11))

    # Fill the training data matrices
    j = 0
    for ix, _ in enumerate(train_data):
        label = train_data[ix]["labels"]
        for feat in train_data[ix]['mspec']:
            X_train[j,:,:] = feat
            y_train[j,:] = label
            j+=1

    # Randomly split training data in a 85% training set and a 15% validation set
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
            test_size=0.15, shuffle=False)

    # Initialize the testing feature and label dictionaries
    X_test = OrderedDict()
    y_test = OrderedDict()

    # Store the number of audio fragments per testing file
    # This will be used for aggrating classification on the test set
    num_fragments_per_file = [len(test_data[i]['mspec']) 
            for i, _ in enumerate(test_data)]

    # Fill the test data dictionaries
    for ix, _ in enumerate(test_data):
        # Initialize the feature and lable matrices for test file at index ix
        X_test_file_ix = np.zeros((num_fragments_per_file[ix], 128, 43))
        y_test_file_ix = np.zeros((num_fragments_per_file[ix], 11))

        label = test_data[ix]["labels"]

        j = 0
        for feat in test_data[ix]['mspec']:
            X_test_file_ix[j,:,:] = feat
            y_test_file_ix[j,:] = label
            j+=1

        X_test[ix] = X_test_file_ix
        y_test[ix] = y_test_file_ix

    # Save Numpy arrays and dictionaries
    np.save('datasets/X_train', X_train)
    np.save('datasets/y_train', y_train)
    np.save('datasets/X_val', X_val)
    np.save('datasets/y_val', y_val)
    
    f = open("datasets/X_test.pkl", "wb")
    pickle.dump(X_test, f)
    f.close()

    f = open("datasets/y_test.pkl", "wb")
    pickle.dump(y_test, f)
    f.close()

