# GPU 
import os 
os.environ["CUDA_DEVICE_ORDER"] = "0"

import pandas as pd
import numpy as np
import cv2
import tensorflow as tf
import keras 

class fasterRCNN:
    def __init__(self, model_name, model_path, class_name, class_path, image_path):
        self.model_name = model_name
        self.model_path = model_path
        self.class_name = class_name
        self.class_path = class_path
        self.image_path = image_path
        self.model = tf.saved_model.load(self.model_path)
        self.class_names = pd.read_csv(self.class_path)
        self.class_names = self.class_names['name'].tolist()


    def prepare_data(self):
        