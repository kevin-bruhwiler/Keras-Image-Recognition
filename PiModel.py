import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import rmsprop, Adam
from keras.callbacks import ModelCheckpoint
import os
from scipy.ndimage import imread
import numpy as np

rescale = 1.0
#Define model
model = Sequential()
model.add(Conv2D(36, 3, 2, activation='relu', input_shape=(194, 256, 3)))
model.add(Conv2D(36, 3, 2, activation='relu'))
model.add(MaxPooling2D(2, strides=2))
model.add(Dropout(0.25))
model.add(Conv2D(36, 3, 2, activation='relu'))
model.add(Conv2D(36, 3, 2, activation='relu'))
model.add(MaxPooling2D(2, strides=2))
model.add(Conv2D(36, 3, 2, activation='relu'))
model.add(Conv2D(36, 3, 2, activation='relu'))
model.add(MaxPooling2D(2, strides=2))
model.add(Flatten())
model.add(Dropout(0.25))
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer=rmsprop(.00001), loss='categorical_crossentropy', metrics=['accuracy'])
print("Model Finished")

x_train = []
y_train = []
x_val = []
y_val = []
PosPicDir = "C:\\Users\\Kevin\\Pictures\\PosPiPics"
NegPicDir = "C:\\Users\\Kevin\\Pictures\\NegPiPics"
posCount = 0
negCount = 0
valPosCount = 0
valNegCount = 0
#Load training data
for file in os.listdir(PosPicDir):
    if np.random.random() < 0.2:
        x_val.append(imread(PosPicDir+"\\"+file))
        valPosCount += 1
    else:
        x_train.append(imread(PosPicDir+"\\"+file))
        posCount += 1
for file in os.listdir(NegPicDir):
    if np.random.random() < 0.2:
        x_val.append(imread(NegPicDir+"\\"+file))
        valNegCount += 1
    else:
        x_train.append(imread(NegPicDir+"\\"+file))
        negCount += 1
x_train = np.asarray(x_train)
x_val = np.asarray(x_val)

#Load labels
for p in range(posCount):
    y_train.append(1) 
for n in range(negCount):
    y_train.append(0)
for pv in range(valPosCount):
    y_val.append(1) 
for nv in range(valNegCount):
    y_val.append(0)    
y_train = keras.utils.to_categorical(y_train,2)
y_val = keras.utils.to_categorical(y_val,2)

modelDir = "C:\\Users\\Kevin\\Documents\\PiModels\\weights.hdf5"
model.fit(x_train, y_train, batch_size=32, epochs=10, verbose=1, 
          callbacks=[ModelCheckpoint(modelDir, verbose=0, save_best_only=True)], validation_data=(x_val, y_val))
