# Custom activation function
from keras.layers import Activation
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
import math

def logsig(x):
    return math.log(K.sigmoid(x))