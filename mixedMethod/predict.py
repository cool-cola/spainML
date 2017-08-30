#coding=utf-8
#Created by spainwang on 2017/7/27.

import os
import h5py
import pandas as pd
import numpy as np
from keras.preprocessing.image import *
from keras.models import *

furture_vec_dir = 'C:\\Users\spainwang\Desktop\Spainwang\\furture_vec'
model_weight_dir = 'C:\\Users\spainwang\Desktop\Spainwang\model_weight\CatVsDog_model_weight.h5'

x_test = []
for filename in os.listdir(furture_vec_dir):
    with h5py.File(furture_vec_dir+'/'+filename, 'r') as h:
        x_test.append(np.array(h['test']))
x_test = np.concatenate(x_test, axis=1)

model = load_model(model_weight_dir)

y_pred = model.predict(x_test, verbose=1)#spainwang:batch_size默认为32
y_pred = y_pred.clip(min=0.005, max=0.995)

print()
#spainwang:
df = pd.read_csv('C:\\Users\spainwang\Desktop\Spainwang\sample_submission.csv')

#spainwang: 可数据扩增
gen = ImageDataGenerator()
test_generator = gen.flow_from_directory('C:\\Users\spainwang\Desktop\Spainwang\\final_test',
                                         (224, 224),
                                         shuffle=False,
                                         batch_size=16,
                                         class_mode=None)
for i, fname in enumerate(test_generator.filenames):
    index = int(fname[fname.rfind('\\')+1:fname.rfind('.')])
    df.set_value(index-1, 'label', y_pred[i])

df.to_csv('C:\\Users\spainwang\Desktop\Spainwang\pred.csv', index=None)
df.head(10)