#coding=utf-8
#Created by Administrator on 2017/8/12.

from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import numpy as np

nb_batch_size = 30
nb_train_samples = 2400
nb_validation_samples = 600

def save_bottleneck_features():
	model = VGG16(include_top=False, weights='imagenet')

	#spainwang: 如何提取bottleneck feature
	#spainwang: 载入图片-图像生成器初始化
	datagen = ImageDataGenerator(rescale=1./255)

	#spainwang: 训练集图像生成器
	train_generator = datagen.flow_from_directory(
    	    'C:\\Users\Administrator\Desktop\data\BlurTest\\train_bulr_clear',
    	    target_size=(150, 150),
    	    batch_size=nb_batch_size,
    	    class_mode=None,
    	    shuffle=False)

	#spainwang:　验证集图像生成器
	val_generator = datagen.flow_from_directory(
    	    'C:\\Users\Administrator\Desktop\data\BlurTest\\validation_blur_clear',
    	    target_size=(150, 150),
    	    batch_size=nb_batch_size,
    	    class_mode=None,
    	    shuffle=False)

	# spainwang:bottleneck feature
	# spainwang:steps是生成器要返回数据的轮数,samples_per_epoch
	bottleneck_features_train = model.predict_generator(train_generator, nb_train_samples//nb_batch_size)
	np.save(open('bottleneck_features_train.npy', 'wb'), bottleneck_features_train)

	bottleneck_features_validation = model.predict_generator(val_generator, nb_validation_samples//nb_batch_size)
	np.save(open('bottleneck_features_validation.npy', 'wb'), bottleneck_features_validation)


def train_top_model():
	# spainwang: 导入bottleneck_features数据
	train_data = np.load(open('bottleneck_features_train.npy', 'rb'))
	train_labels = np.array([0]*int(nb_train_samples/2) + [1]*int(nb_train_samples/2))

	validation_data = np.load(open('bottleneck_features_validation.npy', 'rb'))
	validation_labels = np.array([0]*int(nb_validation_samples/2) + [1]*int(nb_validation_samples/2))

	#spainwang: 设置标签，并规范成Keras默认格式
	model = Sequential()
	model.add(Flatten(input_shape=train_data.shape[1:]))
	model.add(Dense(256, activation='relu'))
	model.add(Dropout(0.5))
	model.add(Dense(2, activation='softmax'))

	#spainwang: 设置参数并训练
	model.compile(loss='sparse_categorical_crossentropy',
				  optimizer='rmsprop',
				  metrics=['accuracy'])

	model.fit(train_data, train_labels,
			  epochs=10,
			  batch_size=nb_batch_size,
			  validation_data=(validation_data, validation_labels))
	model.save_weights('bottleneck_fc_model.h5')

if __name__ == '__main__':
    save_bottleneck_features()  # 图像经过VGG16后保存为bottleneck数据
    train_top_model()  # bottleneck数据用来训练一个处于上层的分类模型