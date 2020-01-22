from keras.models import Sequential 
from keras.layers import Convolution2D , MaxPooling2D , Flatten , Dense , Dropout , Activation  
from keras import backend as K 
import keras 
import os
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2 
import sklearn
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split
from keras.utils import np_utils
K.set_image_dim_ordering('tf')


#1.loading and preprocessing the data

traffic_light='/traffic'
not_traffic_light='/other'
p1=os.listdir(traffic_light))
b1=os.listdir(not_traffic_light)
label=[] #output
i=0

for filename in os.listdir(traffic_light):
        p1[i] =person_1 + '/' + filename
        i=i+1

i=0
for filename in os.listdir(not_traffic_light):
        b1[i] =non_person_1 + '/' + filename
        i=i+1


b1=shuffle(b1,random_state=2)
b1=b1[0:len(p1)]  #the dataset for non traffic light is very large in coparision to traffic light as the classes wont be balanced which makes it difficult for the model to predict smaller class
Data = p1+b1  #input
n=len(Data)
p=len(p1)
labels=np.ones(n,dtype='int64')
labels[0:p]=0
labels[p:n]=1


img_r= 128
img_c=128
channel=1

image=[]
number_of_classess=2


for i in Data:
    img=cv2.imread(i)
    channels = img.shape[2]
    if channels == 3 :
       img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img,(128,128))
    image.append(img)


image=np.array(image)
image=image.astype('float32')
image=image/255.0   #normalisation


#reshaping to feed into cnn
image=np.expand_dims(image, axis=4)    


#convert the class to on-hot encoding
Y=np_utils.to_categorical(labels,number_of_classess)
x,y=shuffle(image,Y,random_state=2)
x_train, x_test , y_train , y_test = train_test_split(x,y,test_size=0.2,random_state=4)



#2.building and training cnn

input_shape=image[0].shape


classifier = Sequential()
classifier.add(Convolution2D(32,(3,3),border_mode='same',input_shape=input_shape))
classifier.add(Activation('relu'))
classifier.add(Convolution2D(32,(3,3))) 
classifier.add(MaxPooling2D(pool_size=(2,2)))
classifier.add(Dropout(0.5))

classifier.add(Flatten())
classifier.add(Dense(128))
classifier.add(Activation('relu'))
classifier.add(Dropout(0.5))  
classifier.add(Dense(number_of_classess))
classifier.add(Activation('softmax'))

classifier.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist=classifier.fit(x_train, y_train, batch_size=32, epochs=25, verbose=1, validation_data=(x_test,y_test))




img1='/home/ananya/Documents/ananya/vision/persons/persons/person_006.bmp'
img1=cv2.imread(img1)
img1=cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
img1=cv2.resize(img1,(128,128))
img1=np.array(img1)
img1=img1.astype('float32')
img1=img1/255.0
img1=np.expand_dims(img1,axis=3)
img1=np.expand_dims(img1,axis=0)
z=classifier.predict_classes(img1)
if z == 0:
   print('traffic light')
if z == 1:
   print('not a traffic light')
