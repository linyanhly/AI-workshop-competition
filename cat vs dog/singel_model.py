import os
import sys
import cv2
import h5py
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.applications import inception_resnet_v2
from keras.applications import Inception_ResNet_V2
from keras.layers import Input, Dense, Dropout, Activation, Flatten, Lambda
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.models import Model
from keras.optimizers import SGD

height = 299
labels = np.array([0] * 11000 + [1] * 11000)
train = np.zeros((22000, height, height, 3), dtype=np.uint8)
test = np.zeros((2000, height, height, 3), dtype=np.uint8)

for i in tqdm(range(11000)):
    img = cv2.imread('train/cat/cat_%s.jpg' % str(i))
    img = cv2.resize(img, (height, height))
    train[i] = img[:, :, ::-1]
    
for i in tqdm(range(11000)):
    img = cv2.imread('train/dog/dog_%s.jpg' % str(i))
    img = cv2.resize(img, (height, height))
    train[i + 11000] = img[:, :, ::-1]

for i in tqdm(range(2000)):
    img = cv2.imread('test/%s.jpg' % str(i))
    img = cv2.resize(img, (height, height))
    test[i] = img[:, :, ::-1]
    
print('Training Data Size = %.2f GB' % (sys.getsizeof(train)/1024**3))
print('Testing Data Size = %.2f GB' % (sys.getsizeof(test)/1024**3))


x = Input(shape=(height, height, 3))
x = Lambda(inception_resnet_v2.preprocess_input)(x)

base_model = Inception_ResNet_V2(include_top=False, input_tensor=x, weights='imagenet', pooling='avg')
train_gap = base_model.predict(train, batch_size=128)
test_gap = base_model.predict(test, batch_size=128)

X_train_gap, X_val_gap, y_train_gap, y_val_gap = train_test_split(train_gap, labels, shuffle=True, test_size=0.01, random_state=42)

x = Input(shape=(X_train_gap.shape[1],))
y = Dropout(0.2)(x)
y = Dense(1, activation='sigmoid', kernel_initializer='he_normal', name='classifier')(y)
model_gap = Model(inputs=x, outputs=y, name='GAP')
model_gap.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])

es = EarlyStopping(monitor='val_loss', patience=20)
mc = ModelCheckpoint(log_dir + log_name, monitor='val_loss', save_best_only=True)
tb = TensorBoard(log_dir=log_dir)
model_gap.fit(x=X_train_gap, y=y_train_gap, batch_size=16, epochs=5, validation_data=(X_val_gap, y_val_gap))
h = model_gap.predict(test_gap).round()
h = pd.DataFrame(h)
h.to_csv('result.csv')
