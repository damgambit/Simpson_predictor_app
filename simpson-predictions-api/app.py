#!/usr/bin/env python
import os
from flask import Flask, request, jsonify
import numpy as np
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import preprocess_input
import pydot
from IPython.display import SVG
from keras.utils.vis_utils import model_to_dot
from keras.utils import plot_model
from keras.initializers import glorot_uniform
from keras.optimizers import Adam
import keras.backend as K
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
from resnets_utils import *
from resnet import *
from utils import decode_img, classes
import json

# Model
K.clear_session()
K.set_image_data_format('channels_last')
K.set_learning_phase(1)
#model = ResNet50(input_shape = (64, 64, 3), classes = 47)
# print(model.summary())
#model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002), 
#               loss='categorical_crossentropy', metrics=['accuracy'])
# model.load_weights('model.h5')
#model.load_weights('best_model.hdf5')

def ResNet50(input_shape = (64, 64, 3), classes = 6):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """
    
    # Define the input as a tensor with shape input_shape
    X_input = Input(input_shape)

    
    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)
    
    # Stage 1
    X = Conv2D(16, (5, 5), strides = (2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis = 3, name = 'bn_conv1')(X)
    X = Activation('relu')(X)
    X = MaxPooling2D((3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f = 3, filters = [16, 16, 64], stage = 2, block='a', s = 1)
    X = identity_block(X, 3, [16, 16, 64], stage=2, block='b')
    X = identity_block(X, 3, [16, 16, 64], stage=2, block='c')


    # Stage 3 
    X = convolutional_block(X, f = 3, filters = [32, 32, 128], stage = 3, block='a', s = 2)
    X = identity_block(X, 3, [32, 32, 128], stage=3, block='b')
    X = identity_block(X, 3, [32, 32, 128], stage=3, block='c')
    X = identity_block(X, 3, [32, 32, 128], stage=3, block='d')


    # Stage 4
    X = convolutional_block(X, f = 3, filters = [64, 64, 256], stage = 4, block='a', s = 2)
    X = identity_block(X, 3, [64, 64, 256], stage=4, block='b')
    X = identity_block(X, 3, [64, 64, 256], stage=4, block='c')
    X = identity_block(X, 3, [64, 64, 256], stage=4, block='d')
    X = identity_block(X, 3, [64, 64, 256], stage=4, block='e')
    X = identity_block(X, 3, [64, 64, 256], stage=4, block='f')

    # Stage 5
    X = convolutional_block(X, f = 3, filters = [128, 128, 512], stage = 5, block='a', s = 2)
    X = identity_block(X, 3, [128, 128, 512], stage=5, block='b')
    X = identity_block(X, 3, [128, 128, 512], stage=5, block='c')

    # AVGPOOL
    X = AveragePooling2D(name='avg_pool')(X)

    # output layer
    X = Flatten()(X)
    X = Dense(classes, activation='softmax', name='fc' + str(classes), kernel_initializer = glorot_uniform(seed=0))(X)
    
    
    # Create model
    model = Model(inputs = X_input, outputs = X, name='ResNet50')

    return model


K.clear_session()

model = ResNet50(input_shape = (64, 64, 3), classes = 47)

print(model.summary())


model.compile(optimizer=Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.002), 
              loss='categorical_crossentropy', metrics=['accuracy'])


model = load_model('best_model.hdf5')


file = 'image3.png'
test_image = image.load_img(file, target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
predicted_class = list(classes.keys())[list(classes.values()).index(result.argmax())]
    
print("The predicted class is:", predicted_class)

default_graph = tf.get_default_graph()

   
# initialization
app = Flask(__name__)


@app.route('/api/predict', methods=['POST'])
def predict():
	print('what is going on?')
	data = request.data
	dataDict = json.loads(data)
	img_base64 = dataDict['image']

	
	img = np.expand_dims(decode_img(img_base64), axis=0)
	img = np.resize(img, (1,64,64,3))
	print('predicting')

	global default_graph
	with default_graph.as_default():
		test_image = image.load_img('to_predict.png', target_size = (64, 64))
		test_image = image.img_to_array(test_image)
		test_image = np.expand_dims(test_image, axis = 0)
		result = model.predict(test_image)
		predicted_class = list(classes.keys())[list(classes.values()).index(result.argmax())]
    
	print(result)
	print("The predicted class is:", predicted_class)

	return jsonify({'prediction': predicted_class})


if __name__ == '__main__': 
    # Bind to PORT if defined, otherwise default to 5000.
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)