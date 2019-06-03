import sys
import pickle
import numpy as np
import progress
from progress.bar import IncrementalBar
from keras.models import load_model

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


def main(argv):
    batch_norm = False
    max_pooling_window = 0
    method = 2
    if len(argv) > 1:
        batch_norm = int(argv[1])
        max_pooling_window = int(argv[2])
        method = int(argv[3])
    print("Max pooling window : " , max_pooling_window)
    print("batch norm :", batch_norm)
    
    if method == 1:
        thetas = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
    elif method == 2:
        thetas = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45,0.50, 0.55, 0.60]
    else:
        print("Wrong argument for method")
    
    ################### Loading the test set and the model ####################
    f = open("datasets/X_test.pkl", "rb")
    X_test = pickle.load(f)
    f.close()

    if batch_norm: model = load_model("results/with_batch_norm/model.h5")
    else: model = load_model("results/without_batch_norm/model.h5")
    ###########################################################################


    ################ Computing the predictions for every theta ################
    predictions = []
    bar = IncrementalBar('Processing', max=9)

    for theta in thetas:
        predictions_theta = np.zeros((len(X_test), 11))
        for (key,val) in X_test.items():
            ### Predicting on the subset
            val = np.reshape(val, (val.shape[0], val.shape[1], val.shape[2], 1))
            prediction = model.predict(val)
    
            ### Class-wise max pooling:
            if max_pooling_window:
                max_pooled_pred = np.zeros_like(prediction)
                N = len(prediction)
                for i in range(N):
                        max_pooled_pred[i] = np.max(prediction[max(0,i-max_pooling_window//2):min(N,i + max_pooling_window//2), :], axis = 0)
                prediction = max_pooled_pred
            
            ### Aggregating the results from the same excerpt
            if method == 1:
                prediction = np.mean(prediction, axis = 0)
            if method == 2:
                prediction = np.sum(prediction, axis = 0)
                m = np.max(prediction)
                prediction /= m
            prediction = prediction - theta
            prediction[prediction > 0] = 1
            prediction[prediction <= 0] = 0
            predictions_theta[key] = prediction
        predictions.append(predictions_theta)
        bar.next()
    bar.finish()
    ###########################################################################  
    
    
    ######################### Saving the predictions ##########################
    if batch_norm:
         pickle.dump(predictions, open("predictions/with" + str(max_pooling_window), "wb"))
    else:
         pickle.dump(predictions, open("predictions/without" + str(max_pooling_window), "wb"))
    ###########################################################################

if __name__ == "__main__":
    main(sys.argv)
