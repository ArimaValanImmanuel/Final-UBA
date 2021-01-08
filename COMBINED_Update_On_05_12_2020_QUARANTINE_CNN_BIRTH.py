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

#os.environ['CUDA_VISIBLE_DEVICES']='2'

EPOCHS = 50
INIT_LR = 1e-3
#to avoid crashing due to less ram
BS = 1000
default_image_size = tuple((150,150))
width=150
height=150
depth=3
inputShape=(150,150,3)

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
  cv_img=cv2.resize(cv_img,default_image_size)
  img = np.uint8(cv_img)
  #img = np.uint8((0.2126 * img[:,:,0]) + np.uint8(0.7152 * img[:,:,1]) + np.uint8(0.0722 * img[:,:,2]))
  #flatten it
  if(flatten):
    img=img.flatten()
  return img
'''    

TRAIN_DIR="C:/Users/arima/OneDrive/Desktop/kaggle_api/tomato/train/"
index=0
data={}
for FOLDER in os.listdir(TRAIN_DIR):
    for image_dir in os.listdir(TRAIN_DIR+FOLDER):
      if index not in data:
        data[index]=[]
      try:  
        data[index].append(img_to_np(TRAIN_DIR+FOLDER+"/"+image_dir))
      except:
        print("Error to load the image "+TRAIN_DIR+FOLDER+"/"+image_dir)
    index=index+1

CLASS_LIMIT=500
colors=["r","b","m","y","k","c","#eeefff","#808000","#4B0082","#CD853F"]
for index_class in range(len(data)):
  index=0
  for arr in data[index_class]:
    plt.hist(arr,color=colors[index_class],alpha=0.5)
    if(index>CLASS_LIMIT):
      plt.title(labels[index_class])
      plt.show()
      break
    index=index+1

input()'''

TRAIN_DIR="C:/Users/arima/OneDrive/Desktop/kaggle_api/tomato/train/"
index=0
data=[]
for FOLDER in os.listdir(TRAIN_DIR):
    print(TRAIN_DIR+FOLDER)
    for image_dir in os.listdir(TRAIN_DIR+FOLDER):
      data.append({"x":img_to_np(TRAIN_DIR+FOLDER+"/"+image_dir,flatten=False),"y":label[index]})
    index=index+1
x,y=[],[]
for obj in data:
  x.append(obj["x"])
  y.append(obj["y"])
x_train = np.array(x,dtype=np.float16)
y_train = np.array(y,dtype=np.float16)

TEST_DIR="C:/Users/arima/OneDrive/Desktop/kaggle_api/tomato/val/"
index=0
data=[]
for FOLDER in os.listdir(TEST_DIR):
    print(TEST_DIR+FOLDER)
    for image_dir in os.listdir(TEST_DIR+FOLDER):
      data.append({"x":img_to_np(TEST_DIR+FOLDER+"/"+image_dir,flatten=False),"y":label[index]})
    index=index+1
x,y=[],[]
for obj in data:
  x.append(obj["x"])
  y.append(obj["y"])
x_test = np.array(x,dtype=np.float16)
y_test = np.array(y,dtype=np.float16)  

print(len(x_train), len(y_train), len(x_test), len(y_test))

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


model.summary()

'''
model.compile(loss="binary_crossentropy", optimizer="adam",metrics=["accuracy"])
                aug = ImageDataGenerator(fill_mode="nearest")
                model_history = model.fit_generator(
        aug.flow(x_train, y_train, batch_size=BS),
        validation_data=(x_test, y_test),
        steps_per_epoch=len(x_train) // BS,
        epochs=EPOCHS, verbose=1
    )
'''

model.compile(loss='binary_crossentropy',#'hinge'
              optimizer='adam',#optimizers.RMSprop(lr=1e-4),
              metrics=['acc'])
#aug=ImageDataGenerator(fill_mode="nearest")

train_datagen = ImageDataGenerator(rescale=1./255,rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,)
val_datagen=ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow(x_train, y_train)
'''((x_train, y_train),
                                      target_size=(150,150),
                                      batch_size=BS,
                                      class_mode='categorical')'''
#label_map = (train_generator.class_indices)

#print(label_map)

val_generator = val_datagen.flow(x_test, y_test)
'''((x_test, y_test),
                                                 target_size=(150,150),
                                                 batch_size=BS,
                                                 class_mode='categorical')'''



history = model.fit_generator(train_generator,
                              validation_data=val_generator,
                              validation_steps=len(x_test)//BS,
                              steps_per_epoch=len(x_train)//BS,
                              epochs=EPOCHS,verbose=1)


'''
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
'''

#input()

#model.save_weights('C:/Users/arima/OneDrive/Desktop/kaggle_api/tomato/weights/model_weights_UPDATE_05_12_2020_fifty_epochs.h5')
model.save('C:/Users/arima/OneDrive/Desktop/ARIMA FOLDER MASTER/PROJECTS/kaggle_api/tomato/weights/model_keras_UPDATE____TWO_____05_12_2020_fifty_epochs.h5')

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

scores = model.evaluate(x_test, y_test)
print("Accuracy is :{}\n\nFinished Ready to Deploy, MASTER!".format(str(scores[1]*100)))


