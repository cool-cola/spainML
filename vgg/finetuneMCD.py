#coding=utf-8
#Created by spainwang on 2017/7/25.
import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras import optimizers
from PIL import ImageFile
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True

# from smallcnn import save_history

def save_history(history, result_file):
    loss = history.history['loss']
    acc = history.history['acc']
    val_loss = history.history['val_loss']
    val_acc = history.history['val_acc']
    nb_epoch = len(acc)
    with open(result_file, "w") as fp:
        fp.write("epoch\tloss\tacc\tval_loss\tval_acc\n")
        for i in range(nb_epoch):
            fp.write("%d\t%f\t%f\t%f\t%f\n" % (i, loss[i], acc[i], val_loss[i], val_acc[i]))


# https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975
img_width, img_height = 210, 280
# top_model_weights_path = 'bottleneck_fc_model' # .h5'
train_data_dir = 'C:\\Users\spainwang\Desktop\\re\\train'  # '../train'
validation_data_dir = 'C:\\Users\spainwang\Desktop\\re\\test'  # '../test'
nb_train_samples = 400  # 2350 #2000
nb_validation_samples = 100  # 150 # 800
nb_epoch = 2  # 8
batch_size = 20  # 10
bottleneck_fc_model = 'C:\\Users\spainwang\Desktop\\re'  # '../bottleneck_fc_model'
result_dir = 'C:\\Users\spainwang\Desktop\\re\\finetune_results'  # '../finetune_results'

if __name__ == '__main__':
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=batch_size,
        class_mode='categorical')

    # https://keras.io/applications/#inceptionv3
    input_tensor = Input(shape=(img_height, img_width, 3))
    vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    # vgg16_model.summary()

    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(5, activation='softmax'))
    top_model.load_weights(os.path.join(bottleneck_fc_model,
                                        'C:\\Users\spainwang\Desktop\\re\\VGG_bottleneck_fc_model98.h5'))  # spainwang:绝对目录C:\\Users\spainwang\Desktop\\re\\

    # https://github.com/fchollet/keras/issues/4040
    model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))

    # Total params: 16,812,353
    # Trainable params: 16,812,353
    # Non-trainable params: 0
    model.summary()  # spainwang: 模型概括打印
    print('-------------------')
    print(len(model.layers))
    for i in range(len(model.layers)):
        print(i, model.layers[i])

    # 冻结部分VGG
    for layer in model.layers[:15]:
        layer.trainable = False

    # Total params: 16,812,353
    # Trainable params: 9,177,089
    # Non-trainable params: 7,635,264
    model.summary()
    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])
    for i in range(1, 3):  # spainwang: 修改前是range（1， 20）
        # Fine-tuning
        history = model.fit_generator(
            train_generator,
            samples_per_epoch=nb_train_samples,
            nb_epoch=nb_epoch,
            validation_data=validation_generator,
            nb_val_samples=nb_validation_samples)
        print("------------111")
        model.save_weights(os.path.join(result_dir, 'finetuning' + str(i * 2) + '.h5'))
        save_history(history, os.path.join(result_dir, 'history_finetuning' + str(i * 2) + '.txt'))
    model.save_weights(os.path.join(result_dir, 'finetuning' + str(i * 2) + '.h5'))
    save_history(history, os.path.join(result_dir, 'history_finetuning' + str(i * 2) + '.txt'))