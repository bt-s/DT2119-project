from keras import applications
from keras import backend as K
import numpy as np
import keras


# load the model
model = keras.models.load_model("../results/without_batch_norm/model.h5")

# get the symbolic outputs of each "key" layer (we gave them unique names).
layer_dict = dict([(layer.name, layer) for layer in model.layers])

# placeholder for the input
input_img = model.inputs[0]

# equivalent to learning rate ...
step = 0.0005
# ... and nr. epochs
N = 1000

# last layer name
layer_name = 'dense_2'

# placeholder of the of the last layer
layer_output = layer_dict[layer_name].output

# load the inputs and the target labels
X = np.load("one_per_instrument.npy")
Y = np.eye(11)

# output matrix where to save the results
X_opt = np.zeros(X.shape)

# for each isntrument
for instr in range(11):

    target = Y[instr,:]
    target_placeholder = K.placeholder(shape=(None,11))

    print("target is :", target)
    print("##########################################")
    
    # build the loss to minimize
    loss = K.categorical_crossentropy(target_placeholder, layer_output)

    # compute the gradient of the input wrt this loss
    grads = K.gradients(loss, input_img)[0]

    # normalization trick: we normalize the gradient
    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5)

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_img], [loss, grads], feed_dict={target_placeholder:target})


    # we start from the given input 
    input_img_data = X[instr,:,:]
    input_img_data = np.reshape(input_img_data,(1,128,43,1))

    # run gradient ascent for N  steps
    for i in range(N):
        loss_value, grads_value = iterate([input_img_data])
       
        pred = model.predict(input_img_data)
        #print(-np.log(pred[0,instr]+np.finfo(float).eps))
        print(loss_value)
        input_img_data -= grads_value * step

    # save result an print the prediction
    X_opt[instr,:,:]= input_img_data[0,:,:,0]

    print(model.predict(input_img_data))

# save the results in a file
np.save("opt_one_per_instr", X_opt)
