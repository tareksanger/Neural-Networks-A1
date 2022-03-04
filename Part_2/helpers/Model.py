from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

class Model(object): 

    def __init__(self, num_hidden_nodes: int):
        self._num_hidden_nodes = num_hidden_nodes

    def __call__(self):
        
        model = Sequential()
        model.add(Dense(self._num_hidden_nodes, activation='sigmoid', input_shape=(7*5, )))
        model.add(Dense(31, activation='sigmoid'))

        # Set Optimizer and Complie    
        optimizer = Adam(learning_rate=0.01)
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        
        return model

        
