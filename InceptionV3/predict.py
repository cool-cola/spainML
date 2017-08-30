#coding=utf-8
#Created by spainwang on 2017/7/25.
from keras.models import load_model
import os
from keras.preprocessing.image import ImageDataGenerator
import numpy as np


img_width = 299
img_height = 299
batch_size = 32
nbr_test_samples = 25#1000

#FishNames = ['ALB', 'BET', 'DOL', 'LAG', 'NoF', 'OTHER', 'SHARK', 'YFT']
ClassNames = ['bus', 'dinosaurs', 'elephants', 'flowers', 'horse']

root_path = 'C:\\Users\Administrator\Desktop\data'#'/Users/pengpai/Desktop/python/DeepLearning/Kaggle/NCFM'

weights_path = 'C:\\Users\Administrator\Desktop\data\model_weight\InceptionV3_kaggle_weight.h5'#os.path.join(root_path, '/model_weight/InceptionV3_kaggle_weight.h5')

test_data_dir = 'C:\\Users\Administrator\Desktop\data\minifortest\\test_data'#os.path.join(root_path, '/minifortest/ftest')

# test data generator for prediction
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
        test_data_dir,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        shuffle = False, # Important !!!
        classes = None,
        class_mode = None)

test_image_list = test_generator.filenames

print('Loading model and weights from training process ...')
InceptionV3_model = load_model(weights_path)

print('Begin to predict for testing data ...')
predictions = InceptionV3_model.predict_generator(test_generator, nbr_test_samples)

np.savetxt(os.path.join(root_path, 'predictions.txt'), predictions)

print('Begin to write submission file ..')
f_submit = open(os.path.join(root_path, 'submit.csv'), 'w')
f_submit.write('image,bus,dinosaurs,elephants,flowers,horse\n')
#f_submit.write('image,ALB,BET,DOL,LAG,NoF,OTHER,SHARK,YFT\n')
for i, image_name in enumerate(test_image_list):
    pred = ['%.6f' % p for p in predictions[i, :]]
    if i % 100 == 0:
        print('{} / {}'.format(i, nbr_test_samples))
    f_submit.write('%s,%s\n' % (os.path.basename(image_name), ','.join(pred)))

f_submit.close()

print('Submission file successfully generated!')
