import os
import numpy as np
import cv2
import time
import random

# import json

LETTERS = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z', 'i_', 'j_', 'n_', 'y_', 'u_']

img_dir = os.path.abspath('Part_2/datasets/Letters')
extension = '.jpg'

TARGET_PATTERNS_X = []
DISTORTED_PATTERNS_X = []
VALIDATION_PATTERNS_X = []

Y = []


DATA = {
    "NOISE_LEVEL_1": {},
    "NOISE_LEVEL_2": {},
    "NOISE_LEVEL_3": {},
}

for l in LETTERS:
    letter = l.replace('_', '')
    Y.append(letter)

    pixel_data = cv2.imread(os.path.join(img_dir, l + extension), flags=0)

    # The images are black and white with the pixel_data written in black
    # When the image is read in all back values are near 0, all white values are near 255
    # Here we convert the values to true and false so that we can easily create our distorted and validation set
    pixel_data[pixel_data<=255//2] = True
    pixel_data[pixel_data>=255//2] = False

    # Create the destorted patterns (For Training)
    flattened_pixel_data = pixel_data.flatten() # Flatten pixel data to easily index the items
    for i in random.sample(range(0, 34), 3):
        flattened_pixel_data[i] = not flattened_pixel_data[i]
    
    DISTORTED_PATTERNS_X.append(np.reshape(flattened_pixel_data, (7,5)).astype(int))

    # Create the Validation Patterns
    noise_level = random.randint(0, 3)
    flattened_pixel_data = pixel_data.flatten() # Flatten pixel data to easily index the items
    for i in random.sample(range(0, 34), noise_level):
        flattened_pixel_data[i] = not flattened_pixel_data[i]
    
    VALIDATION_PATTERNS_X.append(np.reshape(flattened_pixel_data, (7,5)).astype(int))
    
    # Create the images with different levels of noise (For Experiments and Testing)
    for k in range(3):
        flattened_pixel_data = pixel_data.flatten() # Flatten pixel data to easily index the items
        for i in random.sample(range(0, 34), k+1):
            flattened_pixel_data[i] = not flattened_pixel_data[i] 
        
        DATA[f"NOISE_LEVEL_{k+1}"][letter] = np.reshape(flattened_pixel_data, (7,5)).astype(int)
    
    TARGET_PATTERNS_X.append(np.array(pixel_data).astype(int))


np.savez('Part_2/datasets/target_patterns', x=np.array(TARGET_PATTERNS_X), y=np.array(Y))
np.savez('Part_2/datasets/distorted_patterns', x=np.array(DISTORTED_PATTERNS_X), y=np.array(Y))
np.savez('Part_2/datasets/validation_patterns', x=np.array(VALIDATION_PATTERNS_X), y=np.array(Y))

np.savez('Part_2/datasets/noise_level_1', x=np.array(list(DATA['NOISE_LEVEL_1'].values())), y=np.array(list(DATA['NOISE_LEVEL_1'].keys())))
np.savez('Part_2/datasets/noise_level_2',  x=np.array(list(DATA['NOISE_LEVEL_2'].values())), y=np.array(list(DATA['NOISE_LEVEL_2'].keys())))
np.savez('Part_2/datasets/noise_level_3',  x=np.array(list(DATA['NOISE_LEVEL_3'].values())), y=np.array(list(DATA['NOISE_LEVEL_3'].keys())))
