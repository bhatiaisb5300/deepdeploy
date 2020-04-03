import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import cv2
import os
from django.conf import settings
# import seaborn as sns

# from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
# from keras.applications.vgg16 import VGG16
# from tensorflow.python.client import device_lib

# from keras.utils import plot_model
# from tensorflow.keras.models import model_from_json
# from tensorflow.keras.callbacks import TensorBoard
# from tensorflow.keras.callbacks import ModelCheckpoint

# from tensorflow.keras.layers import Dense, Flatten, LSTM, Activation, BatchNormalization
# from tensorflow.keras.layers import Dropout, RepeatVector, TimeDistributed
# from tensorflow.keras import Input, Model
# from tensorflow.keras.layers import Lambda
# from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import model_from_json


def load_model():

	# load json and create model
	json_file = open(os.path.join(settings.BASE_DIR,'deepdeploy/mobilenet/cad_model.json'), 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights("cad_model.h5")

	return loaded_model




# def main(filepath):
# 	# load json and create model
# 	json_file = open('cad_model.json', 'r')
# 	loaded_model_json = json_file.read()
# 	json_file.close()
# 	loaded_model = model_from_json(loaded_model_json)
# 	# load weights into new model
# 	loaded_model.load_weights("cad_model.h5")
# 	# print("Loaded model from disk")
# 	img = plt.imread(filepath)
# 	img = cv2.resize(img, (150,150))
# 	# plt.imshow(img)
# 	# plt.show()
# 	pred = loaded_model.predict(img.reshape(1,150,150,3)/255)
# 	return pred

# if __name__ == "__main__":
# 	filepath = 'cat_or_dog_1.jpg'
# 	pred = main(filepath)
# 	print(pred)
