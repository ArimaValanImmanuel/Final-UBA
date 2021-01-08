import cv2
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
os.environ['CUDA_VISIBLE_DEVICES']='1'
#pprint.pprint(dict(os.environ),width=1)
#print(os.environ['CUDA_VISIBLE_DEVICES'])
#input()
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split as TTS
from keras import layers,models,optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array,load_img
import tensorflow as tf

#os.environ['CUDA_VISIBLE_DEVICES']='2'


model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))#,dtype=int))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(32,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))


#flatten
model.add(layers.Flatten())


#hidden layers
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dropout(0.5))

#output layers
model.add(layers.Dense(10,activation='softmax'))#'sigmoid'

model.compile(loss='categorical_crossentropy',#'hinge'
              optimizer='adam',#optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,)
val_datagen=ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('train',
                                                    target_size=(150,150),
                                                    batch_size=128,
                                                    class_mode='categorical')
label_map = (train_generator.class_indices)

print(label_map)

val_generator = val_datagen.flow_from_directory('val',
                                                 target_size=(150,150),
                                                 batch_size=128,
                                                 class_mode='categorical')

history = model.fit_generator(train_generator,
                              steps_per_epoch=10000//128,
                              epochs=50,
                              validation_data=val_generator,
                              validation_steps=1000//128)

#print(model.evaluate(X_train,y_train),"\n\n",model.evaluate(X_test,y_test))

input()

model.save_weights('C:/Users/arima/OneDrive/Desktop/kaggle_api/tomato/weights/model_weights_UPDATE_05_12_2020_fifty_epochs.h5')
model.save('C:/Users/arima/OneDrive/Desktop/kaggle_api/tomato/weights/model_keras_UPDATE_05_12_2020_fifty_epochs.h5')

acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']

epochs=range(1,len(acc)+1)

#Train and test accuracy
plt.plot(epochs,acc,'b',label='Training Accuracy',color="red")
plt.plot(epochs,val_acc,'b',label='Testing Accuracy')
plt.title("Train and test accuracy")
plt.legend()

plt.figure()

#Train and test loss
plt.plot(epochs,loss,'b',label='Training loss',color="red")
plt.plot(epochs,val_loss,'b',label='Testing loss')
plt.title("Train and test loss")
plt.legend()

plt.show()

input("train, validation finished")
print("models are ready to deploy!")

