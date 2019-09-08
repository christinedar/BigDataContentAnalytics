#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 21:07:55 2019

@author: christinadaramara
"""

### Import packages
from skimage.io import imread
from glob import glob
import IPython.display
import PIL.Image

import math
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from skimage.io import imread
from skimage.measure import block_reduce
from sklearn.metrics import roc_auc_score

from tensorflow.python.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.python.keras.models import Sequential, model_from_json
from tensorflow.python.keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, BatchNormalization, Activation
from keras.layers import Conv2D, MaxPool2D
from keras.optimizers import RMSprop, Adam
from tensorflow.python.lib.io import file_io
import cv2

from sklearn import *

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
import numpy as np 
import keras
import tensorflow as tf

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau

import os
import shutil
from sklearn.utils import shuffle
import gc


#input dir should be changed according to the computer used
INPUT_DIR = '/Users/christinadaramara/Desktop/histopathologic-cancer-detection/'

training_dir = INPUT_DIR + 'train/'
df = pd.DataFrame({'path': glob(os.path.join(training_dir,'*.tif'))})

# the following command should be used for windows
# df['id'] = df.path.map(lambda x: x.split("\\")[1].split(".")[0])
df['id'] = df.path.map(lambda x: x.split("/")[6].split(".")[0]) # number 6 should be replaced with the last path part containing the image label

labels = pd.read_csv(INPUT_DIR + 'train_labels.csv')
# inner join
df_data = df.merge(labels, on = "id")

df_data = shuffle(df_data).reset_index(drop=True)

# Set the id as the index in df_data
#df_data.set_index('id', inplace=True)
#df_data.head()

#read images using the path column
df_data['image'] = df_data['path'].map(cv2.imread)
#crop images and /255
df_data['image'] = df_data['image'].map(lambda x: x[32:64, 32:64]/255.)

#show image 0
from matplotlib import pyplot as plt
plt.imshow(df_data['image'][0])

df_data.columns

#works
fig, ax = plt.subplots(2,5, figsize=(9.5,5))
fig.suptitle('Histopathologic scans of lymph node sections', fontsize=20)    
# Negatives
for i, idx in enumerate(df_data[df_data['label'] == 0].index[:5]):
    ax[0,i].imshow(df_data['image'][idx])
ax[0,0].set_ylabel('Negative samples', size='large')    

for i, idx in enumerate(df_data[df_data['label'] == 1].index[:5]):
    ax[1,i].imshow(df_data['image'][idx])
ax[1,0].set_ylabel('Positive samples', size='large')    


    
df_data['label'][0:5]

ax[0,0].set_ylabel('Negative samples', size='large')
# Positives
for i, idx in enumerate(shuffled_data[shuffled_data['label'] == 1]['id'][:5]):
    path = os.path.join(train_path, idx)
    ax[1,i].imshow(readCroppedImage(path + '.tif'))
ax[1,0].set_ylabel('Tumor tissue samples', size='large')

###############
#data cleansing
###############

# removing this image because it caused a training error previously
df_data = df_data[df_data['id'] != 'dd6dfed324f9fcb6f93f46f32fc800f2ec196be2']

# the following image will result from the following code snipet
# removing this image because it's black
df_data = df_data[df_data['id'] != '9369c7278ec8bcc6c880d99194de09fc2bd4efbe']
df_data.head(3)

# Set the id as the index in df_data
df_data.set_index('id', inplace=True)
df_data.head(1)

#exclude all images very dark or very white   
dark_th = 10 / 255. # If no pixel reaches this threshold, image is considered too dark
bright_th = 245 / 255.  # If no pixel is under this threshold, image is considerd too bright

too_dark_idx = []
too_bright_idx = []

for i in range(0,len(df_data.index)):
    if(df_data['image'][i].max() < dark_th):
            too_dark_idx.append(df_data.index[i])
    if(df_data['image'][i].min() > bright_th):
            too_bright_idx.append(df_data.index[i])
            
#from matplotlib import pyplot as plt
#plt.imshow(df_data[df_data.index == too_dark_idx[0]])
#np.where(df_data.index == too_bright_idx[10])

## Drop images very dark and very bright
df_data.drop(too_dark_idx, inplace=True)
df_data.drop(too_bright_idx, inplace=True)

## Split dataset in train, validation, test

###########################

from sklearn.model_selection import train_test_split
y = df_data['label']
df_train, df_test  = train_test_split(df_data, test_size=0.1, random_state=101, stratify=y)
y = df_train['label']
df_train, df_val = train_test_split(df_train, test_size=0.2, random_state=101, stratify=y)


X_train = np.asarray(df_train['image'].tolist())
X_val = np.asarray(df_val['image'].tolist())
X_test = np.asarray(df_test['image'].tolist())

y_train = np.asarray(df_train['label'].tolist())
y_val = np.asarray(df_val['label'].tolist())
y_test = np.asarray(df_test['label'].tolist())


np.save("Desktop/X_train_array", X_train)
np.save("Desktop/X_val_array", X_val)
np.save("Desktop/y_train_array", y_train)
np.save("Desktop/y_val_array", y_val)
np.save("Desktop/X_test_array", X_test)
np.save("Desktop/y_test_array", y_test)
