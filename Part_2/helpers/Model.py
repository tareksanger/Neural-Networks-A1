from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

# Custom activation function
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import math

# def logsig(x):
#     return math.log(K.sigmoid(x))

class Model(object): 

    def __init__(self, num_hidden_nodes: int):
        self._num_hidden_nodes = num_hidden_nodes

    def __call__(self):

        # logsig activation function
        def logsig(x):
            return K.log(1/(1 + K.exp(-x)))


        get_custom_objects().update({'logsig': Activation(logsig)})


        model = Sequential()
        model.add(Dense(self._num_hidden_nodes, activation='logsig', input_shape=(7*5, )))
        model.add(Dense(31, activation='softmax'))

        # Set Optimizer and Complie    
        optimizer = Adam(learning_rate=0.01)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model

        
