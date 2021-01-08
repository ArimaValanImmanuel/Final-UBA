import cv2
import datetime
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import os,pprint
import random
import gc
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES']='2'
#pprint.pprint(dict(os.environ),width=1)
#print(os.environ['CUDA_VISIBLE_DEVICES'])
#input()
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split as TTS
from keras import layers,models,optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array,load_img
import tensorflow as tf
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.layers.normalization import BatchNormalization
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from sklearn.preprocessing import LabelBinarizer
from keras.models import load_model

#os.environ['CUDA_VISIBLE_DEVICES']='2'
model = load_model('C:/Users/arima/OneDrive/Desktop/ARIMA FOLDER MASTER/PROJECTS/kaggle_api/tomato/weights/Final/model_keras_UPDATE____TWO_____05_12_2020_fifty_epochs.h5')
model.summary()

labels=[
"Tomato___Tomato_mosaic_virus",
"Tomato___Early_blight",
"Tomato___Septoria_leaf_spot",
"Tomato___Bacterial_spot",
"Tomato___Target_Spot",
"Tomato___Spider_mites",
"Tomato___Tomato_Yellow_Leaf_Curl_Virus",
"Tomato___Late_blight",
"Tomato___healthy",
"Tomato___Leaf_Mold"
]

labelencoder = LabelBinarizer()
label=labelencoder.fit_transform([0,1,2,3,4,5,6,7,8,9])

def img_to_np(DIR,flatten=True):
  #canny edge detection by resizing
  cv_img=mpimg.imread(DIR,0)
  cv_img=cv2.resize(cv_img,(150,150))
  img = np.uint8(cv_img)
  #img = np.uint8((0.2126 * img[:,:,0]) + np.uint8(0.7152 * img[:,:,1]) + np.uint8(0.0722 * img[:,:,2]))
  #flatten it
  if(flatten):
    img=img.flatten()
  return img


path="C:/Users/arima/OneDrive/Desktop/ARIMA FOLDER MASTER/PROJECTS/kaggle_api/tomato/real"
files=os.listdir(path)
d=random.choice(files)
arr=img_to_np("{}/{}".format(path,d),flatten=False)
arr=arr.reshape(1,150,150,3)
print(labels[labelencoder.inverse_transform(model.predict(arr))[0]])
