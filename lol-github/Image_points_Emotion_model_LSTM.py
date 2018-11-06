import h5py
import pandas as pd
import os

import numpy as np
import math
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
import keras
from keras.utils import np_utils
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, LSTM, Dense, Dropout, Flatten, Input, concatenate, Conv1D
from keras.layers.convolutional import Conv3D
from keras.layers.core import Permute, Reshape
from keras.layers.convolutional_recurrent import ConvLSTM2D
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping
from keras.layers import Conv2D, MaxPooling2D, ConvLSTM2D
from sklearn.model_selection import train_test_split
from keras.optimizers import Adam, Nadam, Adadelta

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
images = h5py.File('features_images.h5','r')
data_imgs = images['images'][:]

y = pd.read_csv('merged_labels.csv')
emotion_lables = y['Merged Emotion'].values
# integer encode
label_encoder = LabelEncoder()
emotion_lables_encoded = label_encoder.fit_transform(emotion_lables)

clip_lens = pd.read_csv('subclip_lengths.csv')
clip_lengths = np.zeros((614,1))
clip_lengths[0] = 0.822
clip_lengths[1:] = np.reshape(clip_lens['0.822'].values[:613], (613,1))

train_indexes = pd.read_pickle('train_indx.p')
test_indexes = pd.read_pickle('test_indx.p')

def create_LSTM(number_of_classes, feat_dim, WINDOW_SIZE, clip_lens, num_feat_map = 32):

    model.add(BatchNormalization(input_shape=(WINDOW_SIZE,feat_dim)))
    model.add(LSTM(num_feat_map,
                   return_sequences=True))
    model.add(BatchNormalization())
    model.add(LSTM(num_feat_map, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(4,activation='softmax'))

    model.summary()


enc = OneHotEncoder(sparse=False)
emotion_lables_onehot_encoded = enc.fit_transform(np.reshape(emotion_lables_encoded,(-1,1)))
y_train = emotion_lables_onehot_encoded[train_indexes]
y_test = emotion_lables_onehot_encoded[test_indexes]

X_train = data_imgs[train_indexes]
X_test = data_imgs[test_indexes]

clip_lengths_train = clip_lengths[train_indexes]
clip_lengths_test = clip_lengths[test_indexes]

X_train_reshape = np.reshape(X_train, (X_train.shape[0], 25, -1))
X_test_reshape = np.reshape(X_test, (X_test.shape[0], 25, -1))
X_train_reshape.shape
X_test_reshape.shape
feature_size = X_train_reshape.shape[-1]
window_size = X_train_reshape.shape[1]

batch_size = 16
model_patience = 20
epochs = 20
number_of_classes = 4
WINDOW_SIZE = 25
feat_dim = 100*100

## Multi-stream model - Adding clip lengths
clip_len_input = Input(shape=(1,),dtype='float32', name='length')
input = Input(shape=(WINDOW_SIZE,feat_dim), name='X_train')
x = BatchNormalization()(input)
x = LSTM(32, return_sequences=True)(x) # returns a sequence of vectors of dimension 128
x = BatchNormalization()(input)
x = LSTM(32, return_sequences=False)(x)
x = keras.layers.concatenate([x,clip_len_input])
output = Dense(4, activation='softmax')(x)
model = Model(inputs=[input, clip_len_input], outputs=output)
model.summary()
model.compile(
              loss='categorical_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy'])


H = model.fit([X_train_reshape, clip_lengths_train], y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            validation_data=([X_test_reshape, clip_lengths_test], y_test)
            )

preds = model.predict([X_test_reshape,clip_lengths_test])
preds_ = np.argmax(preds, axis = -1)
y_test_ = np.argmax(y_test, axis = -1)

np.savetxt('Results/Image/Static Feature/Image_points_Emotion_predictions_LSTM_ms.txt', preds)
np.savetxt('Results/Image/Static Feature/Image_points_Emotion_LSTM_results_ms.txt', confusion_matrix(y_test_,preds_))
print(classification_report(y_test_,preds_))

# Without static feature
clip_len_input = Input(shape=(1,),dtype='float32', name='length')
input = Input(shape=(WINDOW_SIZE,feat_dim), name='X_train')
x = BatchNormalization()(input)
x = LSTM(32, return_sequences=True)(x) # returns a sequence of vectors of dimension 128
x = BatchNormalization()(input)
x = LSTM(32, return_sequences=False)(x)
output = Dense(4, activation='softmax')(x)
model = Model(inputs=[input, clip_len_input], outputs=output)
model.summary()
model.compile(
              loss='categorical_crossentropy',
              optimizer= 'adam',
              metrics=['accuracy'])


H = model.fit([X_train_reshape, clip_lengths_train], y_train,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            shuffle=True,
            validation_data=([X_test_reshape, clip_lengths_test], y_test)
            )

preds = model.predict([X_test_reshape,clip_lengths_test])
preds_ = np.argmax(preds, axis = -1)
y_test_ = np.argmax(y_test, axis = -1)

np.savetxt('Results/Image/No Static Feature/Image_points_Emotion_predictions_LSTM.txt', preds)
np.savetxt('Results/Image/No Static Feature/Image_points_Emotion_LSTM_results.txt', confusion_matrix(y_test_,preds_))
print(classification_report(y_test_,preds_))
