import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


dataset_path = os.path.abspath("datasets/")

# Load Data
TARGET_PATTERNS = np.load(f"{dataset_path}/target_patterns.npz")
DISTORTED_PATTERNS = np.load(f"{dataset_path}/distorted_patterns.npz")
VALIDATION_PATTERNS = np.load(f"{dataset_path}/validation_patterns.npz")

NOISE_LEVEL_1 = np.load(f"{dataset_path}/noise_level_1.npz")
NOISE_LEVEL_2 = np.load(f"{dataset_path}/noise_level_2.npz")
NOISE_LEVEL_3 = np.load(f"{dataset_path}/noise_level_3.npz")