#!/usr/bin/python3

"""cnn.py: Implementation of the convolutional neural network for predominant
instrument recognition as presented in: 'Deep convolutional neural networks
for predominant instrument recognition in polyphonic music' by Han et al.

Part of a project for the 2019 DT2119 Speech and Speaker Recognition course at
KTH Royal Institute of Technology"""

__author__ = "Pietro Alovisi, Romain Deffayet & Bas Straathof"


import numpy as np

from keras.models import Sequential
from keras.layers import BatchNormalization, Dense, Conv2D, \
        MaxPooling2D, GlobalMaxPooling2D, Dropout, LeakyReLU, ZeroPadding2D

from keras.callbacks import TensorBoard, ModelCheckpoint

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' 

import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)


def create_model(batch_norm=True):
    """Creates the CNN model

    Args:
        batch_norm (bool): decides whether to batch normalize the conv layers

    Returns:
        model (Sequential): CNN model
    """
    # If we apply batch normalization, the bias term should be neglected
    use_bias=True
    if batch_norm:
        use_bias=False
    model = Sequential()

    # Conv block 1: zero padding, 3x3 conv layer with 32 filters,
    # leaky ReLU activation and, optionally, batch_normalization,
    model.add(ZeroPadding2D(padding=(1, 1), input_shape=(1,128,43),
        data_format="channels_first"))
    model.add(Conv2D(32, kernel_size=3, padding='same', use_bias=use_bias,
        data_format='channels_first'))
    model.add(LeakyReLU(alpha=0.33))
    if batch_norm: model.add(BatchNormalization())

    # Conv block 2: zero padding, 3x3 conv layer with 32 filters,
    # leaky ReLU activation and, optionally, batch_normalization,
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_first"))
    model.add(Conv2D(32, kernel_size=3, padding='same', use_bias=use_bias,
        data_format='channels_first'))
    model.add(LeakyReLU(alpha=0.33))
    if batch_norm: model.add(BatchNormalization())

    # 3x3 max pooling layer
    model.add(MaxPooling2D(pool_size=(3, 3), data_format="channels_first"))

    # 0.25 dropout layer
    model.add(Dropout(.25))

    # Conv block 3: zero padding, 3x3 conv layer with 64 filters,
    # leaky ReLU activation and, optionally, batch_normalization,
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_first"))
    model.add(Conv2D(64, kernel_size=3, padding='same', use_bias=use_bias,
        data_format='channels_first'))
    model.add(LeakyReLU(alpha=0.33))
    if batch_norm: model.add(BatchNormalization())

    # Conv block 4: zero padding, 3x3 conv layer with 64 filters,
    # leaky ReLU activation and, optionally, batch_normalization,
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_first"))
    model.add(Conv2D(64, kernel_size=3, padding='same', use_bias=use_bias,
        data_format='channels_first'))
    model.add(LeakyReLU(alpha=0.33))
    if batch_norm: model.add(BatchNormalization())

    # 3x3 max pooling layer
    model.add(MaxPooling2D(pool_size=(3, 3), data_format="channels_first"))

    # 0.25 dropout layer
    model.add(Dropout(.25))

    # Conv block 5: zero padding, 3x3 conv layer with 128 filters,
    # leaky ReLU activation and, optionally, batch_normalization,
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_first"))
    model.add(Conv2D(128, kernel_size=3, padding='same', use_bias=use_bias,
        data_format = 'channels_first'))
    model.add(LeakyReLU(alpha=0.33))
    if batch_norm: model.add(BatchNormalization())

    # Conv block 6: zero padding, 3x3 conv layer with 128 filters,
    # leaky ReLU activation and, optionally, batch_normalization,
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_first"))
    model.add(Conv2D(128, kernel_size=3, padding='same', use_bias=use_bias,
        data_format='channels_first'))
    model.add(LeakyReLU(alpha=0.33))
    if batch_norm: model.add(BatchNormalization())

    # 3x3 max pooling layer
    model.add(MaxPooling2D(pool_size=(3, 3), data_format="channels_first"))

    # 0.25 dropout layer
    model.add(Dropout(.25))

    # Conv block 7: zero padding, 3x3 conv layer with 256 filters,
    # leaky ReLU activation and, optionally, batch_normalization,
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_first"))
    model.add(Conv2D(256, kernel_size=3, padding='same', use_bias=use_bias,
        data_format='channels_first'))
    model.add(LeakyReLU(alpha=0.33))
    if batch_norm: model.add(BatchNormalization())

    # Conv block 8: zero padding, 3x3 conv layer with 256 filters,
    # leaky ReLU activation and, optionally, batch_normalization,
    model.add(ZeroPadding2D(padding=(1, 1), data_format="channels_first"))
    model.add(Conv2D(256, kernel_size=3, padding='same', use_bias=use_bias,
        data_format='channels_first'))
    model.add(LeakyReLU(alpha=0.33))
    if batch_norm: model.add(BatchNormalization())

    # Global max-pooling layer
    model.add(GlobalMaxPooling2D(data_format="channels_first"))

    # 0.25 dropout layer
    model.add(Dropout(.25))

    # Fully connected layer(1024)
    model.add(Dense(1024))

    # 0.50 dropout layer
    model.add(Dropout(.5))

    model.add(Dense(11, activation='sigmoid'))

    model.summary()

    model.compile(optimizer='adam', loss='binary_crossentropy',
            metrics=['accuracy'])

    return model


def train_model(model,  X_train, X_val, y_train, y_val):
    """Trains the CNN model

    Args:
        model   (Sequential): CNN model
        X_train (np.ndarray): input training data
        X_val   (np.ndarray): input val data
        y_train (np.ndarray): input training label
        y_val   (np.ndarray): input val label

    Returns:
        model (Sequential): trained CNN model
    """
    tensorboard = TensorBoard(log_dir='./logs', histogram_freq=0,
                                      write_graph=True, write_images=False)

    checkpoint = ModelCheckpoint('models/model-{epoch:03d}-{acc:03f}-{val_acc:03f}.h5', 
            verbose=1, monitor='val_loss',save_best_only=True, mode='auto') 
    callbacks = [tensorboard, checkpoint]
    
    model.fit(x=X_train, y=y_train, batch_size=128, epochs=20,
            validation_data=(X_val, y_val), callbacks=callbacks)

    return model


def test_model(model, X_test, y_test):
    """Tests the CNN model

    Args:
        model   (Sequential): CNN model
        X_test (np.ndarray): input testing data
        y_test (np.ndarray): input testing label
    """
    prediction = model.predict(X_test)

    print(prediction.shape, y_test.shape)
    prediction = prediction - 0.5

    prediction[prediction > 0] = 1
    prediction[prediction <= 0] = 0

    print(np.sum(prediction == y_test)/y_test.shape[0])
    print(y_test.shape)


if __name__ == "__main__":
    data = np.load("datasets.npz")

    X_train = np.load("datasets/X_train.npy") 
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1],
        X_train.shape[2]))
    y_train = np.load("datasets/y_train.npy")

    X_val = np.load("datasets/X_val.npy")
    X_val = np.reshape(X_val, (X_val.shape[0], 1, X_val.shape[1],
        X_val.shape[2]))
    y_val = np.load("datasets/y_val.npy")

    model = create_model(batch_norm=True)
    model = train_model(model, X_train, X_val, y_train, y_val)

