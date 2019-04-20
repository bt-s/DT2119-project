import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D, GlobalMaxPooling2D, Dropout

from data import loadData

def create_model():
    model = Sequential()
    
    model.add(Conv2D(32, kernel_size = 3, activation='relu', padding='same', data_format = 'channels_first', input_shape = (1,43,128)))
    model.add(Conv2D(32, kernel_size = 3, activation='relu', padding='same', data_format = 'channels_first'))
    model.add(MaxPooling2D(3))
    model.add(Dropout(.25))
    
    
    model.add(Conv2D(64, kernel_size = 3, activation='relu', padding='same', data_format = 'channels_first'))
    model.add(Conv2D(64, kernel_size = 3, activation='relu', padding='same', data_format = 'channels_first'))
    model.add(MaxPooling2D(3))
    model.add(Dropout(.25))
    
    
    model.add(Conv2D(128, kernel_size = 3, activation='relu', padding='same', data_format = 'channels_first'))
    model.add(Conv2D(128, kernel_size = 3, activation='relu', padding='same', data_format = 'channels_first'))
    model.add(MaxPooling2D(3))
    model.add(Dropout(.25))
    
    
    model.add(Conv2D(256, kernel_size = 3, activation='relu', padding='same', data_format = 'channels_first'))
    model.add(Conv2D(256, kernel_size = 3, activation='relu', padding='same', data_format = 'channels_first'))
    model.add(GlobalMaxPooling2D())
    model.add(Dropout(.25))
    
    
    model.add(Dense(1024))
    model.add(Dropout(.25))
    model.add(Dense(11, activation = 'sigmoid'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    
    return model


def train_model(model,  X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = 3)


if __name__ == "__main__":
    #X_train, X_test, y_train, y_test = loadData()
    X_train = np.zeros((1,1,43,128))
    y_train = np.zeros((1,11))
    y_train[0,0] = 1
    X_test = np.zeros((1,1,43,128))
    y_test = np.zeros((1,11))
    y_test[0,0] = 1
    model = create_model()
    train_model(model, X_train, X_test, y_train, y_test)