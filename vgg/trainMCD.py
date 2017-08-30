#coding=utf-8
#Created by spainwang on 2017/7/25.
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from PIL import ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
# dimensions of our images.
img_width, img_height = 210, 280

top_model_weights_path = 'C:\\Users\spainwang\Desktop\\re\VGG_bottleneck_fc_model'  # '../bottleneck_fc_model/bottleneck_fc_model' # .h5'
train_data_dir = 'C:\\Users\spainwang\Desktop\\re\data\\bananaAndhamburg\\fvalid\\train'  # 'C:\\Users\spainwang\Desktop\\re\\train'
validation_data_dir = 'C:\\Users\spainwang\Desktop\\re\data\\bananaAndhamburg\\fvalid\\test'  # 'C:\\Users\spainwang\Desktop\\re\\test'
nb_train_samples = 2774  # 400#2350#230 #2000
nb_validation_samples = 694  # 100#150#30 # 800
epochs = 2
batch_size = 12  # 10


def save_bottlebeck_features():
    datagen = ImageDataGenerator(rescale=1. / 255)

    # build the VGG16 network
    model = applications.VGG16(include_top=False, weights='imagenet')

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print('to run train samples on VGG16')
    bottleneck_features_train = model.predict_generator(
        generator, nb_train_samples // batch_size)
    print('have run train samples on VGG16' + str(bottleneck_features_train.shape))
    np.save(open('C:\\Users\spainwang\Desktop\\re\VGG_bottleneck_fc_model\\bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)

    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    print('to run validation samples on VGG16')
    bottleneck_features_validation = model.predict_generator(
        generator, nb_validation_samples // batch_size)
    print('have run validation samples on VGG16' + str(bottleneck_features_validation.shape))
    np.save(open('C:\\Users\spainwang\Desktop\\re\VGG_bottleneck_fc_model\\bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)
    print('Done save_bottlebeck_features!')


def train_top_model():
    # 读取bottleneck
    train_data = np.load(
        open('C:\\Users\spainwang\Desktop\\re\VGG_bottleneck_fc_model\\bottleneck_features_train.npy', 'rb'))
    print('read bottleneck_features_train.npy ' + str(train_data.shape))
    train_labels = np.array(
        [0] * int(nb_train_samples / 4) + [1] * int(nb_train_samples / 4) + [2] * int(nb_train_samples / 4) + [3] * int(
            nb_train_samples / 4))

    print(len(train_data), len(train_labels))
    validation_data = np.load(
        open('C:\\Users\spainwang\Desktop\\re\VGG_bottleneck_fc_model\\bottleneck_features_validation.npy', 'rb'))
    print('read bottleneck_features_validation.npy ' + str(validation_data.shape))
    # validation_labels = np.array(
    #     [0] * int(nb_validation_samples / 3) + [1] * int(nb_validation_samples / 3)+[2]*int(nb_validation_samples / 3))
    validation_labels = np.array(
        [0] * int(nb_validation_samples / 4) + [1] * int(nb_validation_samples / 4) + [2] * int(
            nb_validation_samples / 4) + [3] * int(nb_validation_samples / 4))

    print(len(validation_data), len(validation_labels))
    print(validation_labels.shape)
    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4, activation='softmax'))
    model.compile(optimizer='rmsprop',
                  loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    for i in range(50):
        model.fit(train_data, train_labels,
                  epochs=epochs,
                  batch_size=batch_size,
                  validation_data=(validation_data, validation_labels))
        model.save_weights(top_model_weights_path + str(i * 2) + '.h5')


if __name__ == '__main__':
    save_bottlebeck_features()  # 图像经过VGG16后保存为bottleneck数据
    train_top_model()  # bottleneck数据用来训练一个处于上层的分类模型