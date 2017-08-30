#coding=utf-8
#Created by spainwang on 2017/7/25.
import os
from keras.applications.vgg16 import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Input, Activation, Dropout, Flatten, Dense
from keras import optimizers
import numpy as np
#from smallcnn import save_history
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
# https://gist.github.com/fchollet/7eb39b44eb9e16e59632d25fb3119975

img_width, img_height = 210, 280

result_dir = 'C:\\Users\spainwang\Desktop\\re\\finetune_results'#'../finetune_results'
def loadModel():
    # https://keras.io/applications/#inceptionv3
    input_tensor = Input(shape=(img_height, img_width, 3))
    vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)
    #vgg16_model.summary()

    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(5, activation='softmax'))

    # top_model.load_weights(os.path.join(result_dir, 'bottleneck_fc_model96.h5'))

    # https://github.com/fchollet/keras/issues/4040
    model = Model(input=vgg16_model.input, output=top_model(vgg16_model.output))
    print('vgg16_model:', vgg16_model)
    print('top_model:', top_model)
    print('model:', model)

    # Total params: 16,812,353
    # Trainable params: 16,812,353
    # Non-trainable params: 0
    model.summary()

    # for i in range(len(model.layers)):
    #     print(i, model.layers[i])

    # for layer in model.layers[:15]:
    #     layer.trainable = False

    # Total params: 16,812,353
    # Trainable params: 9,177,089
    # Non-trainable params: 7,635,264
    model.summary()

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
                  metrics=['accuracy'])

    #model.load_weights(os.path.join(result_dir, 'finetuning4.h5'))
    model.load_weights(os.path.join( 'C:\\Users\spainwang\Desktop\\re\\finetune_results\\finetuning4.h5'))
    return model

def runPred(test_dir,model):
    filenames = os.listdir(test_dir)

    countNone = 0
    countMCD  = 0
    countCard = 0
    print ('============Pred imgs in folder: ', test_dir, '============')
    for fname in filenames:
        img = load_img(test_dir+'/'+fname)  # this is a PIL image
        img = img.resize( (img_width, img_height) )
        x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
        x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
        x = x / 255.0
        #print ('img size: ', x.shape)
        y_pred = model.predict(x)
        # print("概率：")
        print(y_pred[0])
        maxposition=np.argmax(y_pred[0])
        #print(maxposition)
        if(maxposition==0):
            countMCD += 1
            text = 'bus'#'小黄人面包'
        elif(maxposition==1):
            countNone += 1
            text = 'dinosaurs'#'啥也不是'
        elif (maxposition==2):
            countCard+=1
            text = 'elephants'#'名片'
        elif(maxposition==3):
            countMCD += 1
            text = 'flowers'#'快递单'
        else:
            countNone+=1
            text = 'horse'#'题目'

        print(fname + '--%s ' % text + '\n')

def runPic(img,model):
    img = img.resize((img_width, img_height))
    x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
    x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
    x = x / 255.0
    # print ('img size: ', x.shape)
    y_pred = model.predict(x)

    maxposition = np.argmax(y_pred[0])
    # print(maxposition)
    if (maxposition == 0):

        result = 'bus'#'MCD'
    elif (maxposition == 1):

        result = 'dinosaurs'#'None'
    elif(maxposition==2):

        result = 'elephants'#'businesscard'
    elif(maxposition==3):
        result = 'flowers'#'expressorder'
    else:
        result = 'horse'#'topic'

    return result


if __name__ == '__main__':

    model=loadModel()

    # test_dir = 'C:\\Users\spainwang\Desktop\\re\\test\\bus'#'../test/名片'
    # runPred(test_dir,model)
    # test_dir = 'C:\\Users\spainwang\Desktop\\re\\test\dinosaurs'#'../test/快递单'
    # runPred(test_dir,model)
    # test_dir= 'C:\\Users\spainwang\Desktop\\re\\test\elephants'#'../test/题目'
    # runPred(test_dir,model)
    # test_dir = 'C:\\Users\spainwang\Desktop\\re\\test\\flowers'#'../test/all_negatives_samples'
    # runPred(test_dir, model)
    # test_dir = 'C:\\Users\spainwang\Desktop\\re\\test\horse'#'../test/Minion_Burger_bgs'
    # runPred(test_dir, model)

    #runPic(img,model)
    #img = 'C:\\Users\spainwang\Desktop\\re\\test\horse\\700.jpg'#'../test/all_negatives_samples/IMG_0567.JPG'
    test_dir = 'C:\\Users\spainwang\Desktop\\re\\test\\test'
    print(runPred(test_dir, model))
    # for img in os.listdir(test_dir):
    #     im = load_img(test_dir+'/'+img)
    #     print(runPic(im, model))


    # im = load_img(img)  # this is a PIL image
    # print(runPic(im, model))


