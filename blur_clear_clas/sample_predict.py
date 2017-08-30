#coding=utf-8
#Created by Administrator on 2017/8/13.

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
import os
import numpy as np

weight = 150
height = 150

model = Sequential()
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

model.load_weights('blur_clear_weight_model_categorical.h5')

Index = 0
Count_blur = 0
Count_clear = 0
test_dir = 'C:\\Users\Administrator\Desktop\data\BlurTest\\test_pics'
for img in os.listdir(test_dir):
	Img = load_img(test_dir+'/'+img)
	Img = Img.resize( (weight, height) )
	x = img_to_array(Img)
	x = x.reshape((1,) + x.shape)  # this is a Numpy array
	x = x / 255.0
	Index = Index+1
	print(Index)
	preds = model.predict(x)
	maxposition = np.argmax(preds[0])
	if (maxposition == 0):
		print('blur_picture', img, preds[0])
		Count_blur += 1
	else:
		print('clear_picture', img, preds[0])
		Count_clear += 1

print('clear_picture:', Count_clear, 'clear: ', Count_clear/Index)
print('blur_picture:', Count_blur,'blur: ', Count_blur/Index)