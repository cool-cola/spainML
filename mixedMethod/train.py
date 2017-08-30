#coding=utf-8
#Created by spainwang on 2017/7/27.

import h5py
import os
import numpy as np
from sklearn.utils import shuffle
from keras.models import *
from keras.layers import *

np.random.seed(2017)

furture_vec_dir = './furture_vec'
batch_sizes = 10#128
epochs = 8
validation_splits = 0.2
nb_classes = 4

x_train = []

for filename in os.listdir(furture_vec_dir):
    with h5py.File(furture_vec_dir + '/' + filename, 'r') as h:
        x_train.append(np.array(h['train']))
        y_train = np.array(h['label'])

x_train = np.concatenate(x_train, axis=1)#spainwang:组合函数
x_train, y_train = shuffle(x_train, y_train)

#spainwang:construct the model
input_tensor = Input(x_train.shape[1:])
x = Dropout(0.5)(input_tensor)
#x = Dense(nb_classes, activation='sigmoid')(x) #spainwang: 二分类时nb_classes=1
x = Dense(nb_classes, activation='softmax')(x)
model = Model(input_tensor, x)

model.summary()

#spainwang: 修改此处参数，为多分类，现在是二分类模型
#model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])
model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=batch_sizes, epochs=epochs, validation_split=validation_splits)

#save model
model.save('model_weight.h5')