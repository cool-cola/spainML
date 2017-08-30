#coding=utf-8
#Created by spainwang on 2017/7/26.

from keras.models import *
from keras.layers import *
from keras.applications import *
from keras.preprocessing.image import *

import h5py

nb_train_samples = 400#25000
nb_test_samples = 100#12500
batch_size = 10#16

train_dir = 'C:\\Users\Administrator\Desktop\data\minifortest\\train'
test_dir = 'C:\\Users\Administrator\Desktop\data\minifortest\\validation_data'

def write_furture(MODEL, image_size, lambda_func=None):
    width = image_size[0]
    height = image_size[1]
    input_tensor = Input((height, width, 3))
    x = input_tensor
    if lambda_func:
        x = Lambda(lambda_func)(x)
    base_model = MODEL(input_tensor = x, weights = 'imagenet', include_top = False)
    #spainwang：这之间可以对base_model添加部分全连接，进行finetune
    model = Model(base_model.input, GlobalAveragePooling2D()(base_model.output))

    #spainwang：此处可以做数据扩增，但这儿什么数据扩增操作都没做
    gen = ImageDataGenerator()
    train_generator = gen.flow_from_directory(train_dir,
                                              target_size=image_size,
                                              shuffle=False,
                                              batch_size=batch_size)
    test_generator = gen.flow_from_directory(test_dir,
                                             target_size=image_size,
                                             shuffle=False,
                                             batch_size=batch_size,
                                             class_mode=None)

    train = model.predict_generator(train_generator, nb_train_samples//batch_size)
    test = model.predict_generator(test_generator, nb_test_samples//batch_size)
    with h5py.File('furture_%s.h'%MODEL.__name__) as h:
        h.create_dataset('train', data=train)
        h.create_dataset('test', data=test)
        h.create_dataset('label', data=train_generator.classes)

#spainwang: 可自主选择网络模型
write_furture(ResNet50, (224, 224))
write_furture(Xception, (299, 299), xception.preprocess_input)
write_furture(InceptionV3, (299, 299), inception_v3.preprocess_input)
write_furture(VGG16, (224, 224))
# write_furture(VGG19, (224, 224))



