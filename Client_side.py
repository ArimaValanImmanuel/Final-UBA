import cv2
import datetime
import os
import random
import gc

'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split as TTS
from keras import layers,models,optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array,load_img
import tensorflow as tf
from keras.models import load_model
'''

os.environ['CUDA_VISIBLE_DEVICES']='1'

faceDetect = cv2.CascadeClassifier('C:/Users/arima/OneDrive/Desktop/ARIMA FOLDER MASTER/PROJECTS/kaggle_api/opencv-android/OpenCV-android-sdk/sdk/etc/haarcascades/haarcascade_smile.xml')#frontalface_alt_tree.xml')#default.xml')

video = cv2.VideoCapture(0)

a=0

while True:

    a=a+1

    now=datetime.datetime.now()
    
    check,frame=video.read()

    print("this is check",check)
    print("this is frame",frame)

    faces = faceDetect.detectMultiScale(frame,1.3,5);

    for(x,y,w,h) in faces:
       cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

    name = str(now.strftime("%Y-%m-%d %H:%M:%S"))
    cv2.imshow("",frame)
    cv2.imwrite("C:/Users/arima/OneDrive/Desktop/ARIMA FOLDER MASTER/PROJECTS/kaggle_api/tomato/real/I{}.png".format(a),frame)

    key=cv2.waitKey(1)

    if key == ord('q'):
        cv2.destroyAllWindows()

        break

print(a)                 

video.release()

#cv2.destroyAllWindows()
