import os
import numpy as np
import pandas as pd
from sklearn.utils import shuffle
from helpers.constants import TARGET_PATTERNS, DISTORTED_PATTERNS, VALIDATION_PATTERNS
from helpers.Model import Model

save_dir = os.path.abspath('Saves/')

# Each Character is mapped to a numeric value between 0 and 30
# [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30]
# ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'i', 'j', 'n', 'y', 'u']
y_true = np.array(list(range(31)))

# Flatten Each of the datasets
target_patterns_flattened_x = np.array(TARGET_PATTERNS['x']).reshape(-1, 7 * 5)         # Target Patterns
distorted_patterns_flattened_x = np.array(DISTORTED_PATTERNS['x']).reshape(-1, 7 * 5)   # Distorted Patterns
val_patterns_flattend_x = np.array(VALIDATION_PATTERNS['x']).reshape(-1, 7 * 5)         # Validation Patterns

# Shuffle the Images 
target_patterns_flattened_x, target_patterns_y = shuffle(target_patterns_flattened_x, y_true)
distorted_patterns_flattened_x, distorted_patterns_y = shuffle(distorted_patterns_flattened_x, y_true)
val_patterns_flattend_x, val_patterns_y = shuffle(val_patterns_flattend_x, y_true)

# Create and Train Models for each the various number of hidden neurons (5, 10, 15, 20, 25).
for neurons in range(5, 30, 5):
    # Create the Model
    model = Model(num_hidden_nodes=neurons)()
    # Display the Summary
    model.summary()

    # Step 1: Train the Model on the noise free data
    history_1 = model.fit(target_patterns_flattened_x, target_patterns_y, epochs=300, validation_data=(val_patterns_flattend_x, val_patterns_y), batch_size=1)

    model.save(os.path.join(save_dir, f'{neurons}/noise_free_model.h5'))
    model.save_weights(os.path.join(save_dir, f'{neurons}/noise_free_final_weights.hdf5'), overwrite=True)

    # Step 2: Train the Model on the Distorted data (Level 3 noise)
    history_2 = model.fit(distorted_patterns_flattened_x, distorted_patterns_y, epochs=600, validation_data=(val_patterns_flattend_x, val_patterns_y), batch_size=1, initial_epoch=301)
    # Step 3: Retrain the data on the noise free data
    history_3 = model.fit(target_patterns_flattened_x, target_patterns_y, epochs=900, validation_data=(val_patterns_flattend_x, val_patterns_y), batch_size=1, initial_epoch=601)

    # Combine the history from each step
    history = pd.concat([
        pd.DataFrame(history_1.history),
        pd.DataFrame(history_2.history),
        pd.DataFrame(history_3.history)
    ],
    ignore_index=True
    )
    
    # Save History, Model and Weights
    history.to_hdf(os.path.join(save_dir, f"{neurons}/history.h5"), 'history')
    model.save(os.path.join(save_dir, f'{neurons}/model.h5'))
    model.save_weights(os.path.join(save_dir, f'{neurons}/final_weights.hdf5'), overwrite=True)
