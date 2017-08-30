#coding=utf-8
#Created by Administrator on 2017/8/11.

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
import os
import numpy as np

weight = 150
height = 150
epoches = 10
nb_train_samples = 2400
nb_validation_samples = 600

model = Sequential()
#spainwang: tensorflow is channels_last for input_shape
model.add(Conv2D(32, (3, 3), input_shape=(weight, height, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) #spainwang: 3d to 1d
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2)) # spainwang: class blur or clear pic
model.add(Activation('softmax'))

#spainwang: 二分类，多分类loss=‘categorical_crossentropy’
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

#图像预处理
train_datagen = ImageDataGenerator(rescale=1./255,
								   shear_range=0.2,
								   zoom_range=0.2,
								   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory('C:\\Users\Administrator\Desktop\data\BlurTest\\train_bulr_clear',
													target_size=(weight, height),
													batch_size=32,
													class_mode='categorical') #spainwang: 多分类 class_mode=categorical

val_generator = test_datagen.flow_from_directory('C:\\Users\Administrator\Desktop\data\BlurTest\\validation_blur_clear',
												 target_size=(weight, height),
												 batch_size=32,
												 class_mode='categorical')

model.summary()
#spainwang: training the model
model.fit_generator(train_generator,
					samples_per_epoch=nb_train_samples,
					nb_epoch=epoches,
					validation_data=val_generator,
					nb_val_samples=nb_validation_samples)

model.save_weights('blur_clear_weight_model_categorical.h5')