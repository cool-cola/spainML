#coding=utf-8
#Created by Administrator on 2017/8/13.

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential, Model
from keras.layers import Input, Dropout, Flatten, Dense
from keras.applications.vgg16 import VGG16

batch_size = 32
weight = 150
height = 150
epochs = 10
nb_train_samples = 2400
nb_validation_samples = 600

input_tensor = Input(shape=(weight, height, 3))
model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor)

top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(2, activation='softmax'))

#spainwang：top_model_weights_path
top_model.load_weights('bottleneck_fc_model.h5')

#spainwang: challen has to do like this
model = Model(inputs=model.input, outputs=top_model(model.output))

model.summary()

#spainwang: 冻结网络的一部分参数
for layer in model.layers[:15]:
	layer.trainable = False

model.summary()

model.compile(loss='categorical_crossentropy',
			  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
			  metrics=['accuracy'])

#spainwang: 图片预处理生成器
train_datagen = ImageDataGenerator(rescale=1./255,
								   shear_range=0.2,
								   zoom_range=0.2,
								   horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

#spainwang: 图片generator
train_generator = train_datagen.flow_from_directory('C:\\Users\Administrator\Desktop\data\BlurTest\\train_bulr_clear',
													target_size=(weight, height),
													batch_size=batch_size,
													class_mode='categorical')

validation_generator = test_datagen.flow_from_directory('C:\\Users\Administrator\Desktop\data\BlurTest\\validation_blur_clear',
													target_size=(weight, height),
													batch_size=batch_size,
													class_mode='categorical')

#spainwang: 训练
model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
model.save_weights('finetuning_result_weight.h5')








