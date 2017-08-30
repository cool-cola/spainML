#coding=utf-8
#Created by Administrator on 2017/8/14.

from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator,img_to_array, load_img
import os
import numpy as np
from keras import optimizers

weight = 150
height = 150

def loadModel():
	input_tensor = Input(shape=(weight, height, 3))
	model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

	top_model = Sequential()
	top_model.add(Flatten(input_shape=model.output_shape[1:]))
	top_model.add(Dense(256, activation='relu'))
	top_model.add(Dropout(0.5))
	top_model.add(Dense(2, activation='softmax'))

	model = Model(inputs=model.input, outputs=top_model(model.output))

	model.load_weights('finetuning_result_weight.h5')
	return model

def predict(test_dir, model):
	Count_blur = 0
	Count_clear = 0
	for img in os.listdir(test_dir):
		Img = load_img(test_dir+'/'+img)
		Img = Img.resize((weight, height))
		x = img_to_array(Img)
		x = x.reshape((1,)+x.shape)
		x = x/255.0
		preds = model.predict(x)

		maxposition = np.argmax(preds[0])
		if (maxposition == 0):
			print('blur_picture', img, preds[0])
			Count_blur += 1
		else:
			print('clear_picture', img, preds[0])
			Count_clear += 1

	return Count_blur, Count_clear

if __name__ == '__main__':
	model = loadModel();

	test_dir = 'C:\\Users\Administrator\Desktop\data\BlurTest\\test_pics'
	Count_blur, Count_clear = predict(test_dir, model)

	Index = Count_clear + Count_blur
	print('clear_picture:', Count_clear, 'clear: ', Count_clear / Index)
	print('blur_picture:', Count_blur, 'blur: ', Count_blur / Index)