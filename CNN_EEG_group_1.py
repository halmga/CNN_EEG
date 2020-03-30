# Team SpeechBot

import io
import pickle
from keras.utils import to_categorical
from keras.utils import np_utils
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras import Sequential
from tensorflow.keras import layers
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, Flatten, Dense, Dropout
from tensorflow.keras.layers import MaxPooling2D, Conv3D, MaxPooling3D, BatchNormalization, AveragePooling2D
from kapre.utils import Normalization2D
import numpy as np
from keras.models import load_model
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import tensorflow  as tf
from keras.layers import *
from keras.regularizers import l2
from keras import optimizers
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from keras.layers import RepeatVector, SimpleRNN, ELU
from keras.layers import DepthwiseConv2D
from keras.layers import SeparableConv2D

#Data input
from google.colab import drive
drive.mount('/content/drive')

#Spectrograms, RAW data, Test set input
infile = open('/content/drive/My Drive/Deep_Learning/Data_Spectrograms.pkl','rb')
#comb_features = open('/content/drive/My Drive/Deep_Learning/combined_feats.npy','rb')
infileraw = open('/content/drive/My Drive/Deep_Learning/Data_Raw_signals.pkl','rb')
spec_f = pickle.load(infile)
#comb_feat = np.load(comb_features)
raw_f = pickle.load(infileraw)
test_input = open('/content/drive/My Drive/Deep_Learning/Test_Spectrograms_no_labels.pkl','rb')
test_f = pickle.load(test_input)

#Activation Function SWISH
from keras.backend import sigmoid
from keras.utils.generic_utils import get_custom_objects
from keras.layers import Activation

def swish(x, beta = 1):
    return (x * sigmoid(beta * x))

unique, counts = np.unique(spec_f[1], return_counts=True)
dict(zip(unique, counts))
#Used SMOTE - Synthetic Minority Over-sampling Technique to balance classes
train_data = spec_f 


x_train = train_data[0] 
y_labels = train_data[1]

x_train = x_train.reshape((15375,6000))

smote = SMOTE('auto')
train_images_sm, test_labels_sm = smote.fit_sample(x_train, y_labels)

print(train_images_sm.shape)
unique, counts = np.unique(test_labels_sm, return_counts=True)
dict(zip(unique, counts))

#SVMSMOTE LOAD

SVMSMOTE_images = open('/content/drive/My Drive/Deep_Learning/SVMSMOTE_images.npy','rb')
SVMSMOTE_labels = open('/content/drive/My Drive/Deep_Learning/SVMSMOTE_labels.npy','rb')

svm_img = np.load(SVMSMOTE_images)
svm_lable = np.load(SVMSMOTE_labels)

svm_lable.shape

#################
# FINAL VERSION # + EXTRAS
#################


#EEGNet 89% L2 regularizer
#UPLOADED MODEL 70.6% acc on challenge

#Simple version of the net
#SWISH + 1 extra dense in bwtwee  conv layers

x_train = train_images_sm
y_labels = test_labels_sm

x_test = test_f


x_test =x_test[0].reshape (1754,100,30,2)

x_train= x_train.reshape(18000,100, 30, 2 )

train_images, test_images, train_labels,  test_labels = train_test_split(x_train, y_labels, 
                                                                         test_size = 0.01, 
                                                                         random_state = 666)

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

early_stop = EarlyStopping(patience=5)
l2_val = 0.00012
ad_g = optimizers.Adagrad(lr=0.001)
elu = keras.layers.ELU(alpha=2.0)


model = Sequential()
model.add(Conv2D(64, (5, 1), activation= 'tanh', input_shape=(100, 30, 2),padding="same"
                 ,kernel_initializer='he_normal',kernel_regularizer=l2(l2_val),
                 bias_regularizer=l2(l2_val)))
model.add(BatchNormalization())
model.add(Activation(elu))
model.add(DepthwiseConv2D((2,1), strides=(1, 1), padding='same', depth_multiplier=4, data_format=None, 
                          dilation_rate=(1, 1), activation= None, use_bias=True, 
                          depthwise_initializer='glorot_uniform', bias_initializer='zeros', 
                          depthwise_regularizer=None, bias_regularizer=None, activity_regularizer=None, 
                          depthwise_constraint=None, bias_constraint=None))
model.add(BatchNormalization())
model.add(Activation(elu))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))

model.add(Dropout(0.5))
model.add(SeparableConv2D(16, (1,16), strides=(1, 1), padding='same', data_format=None, dilation_rate=(1, 1), 
                          depth_multiplier=8, activation= None, use_bias=True, 
                          depthwise_initializer='glorot_uniform', pointwise_initializer='glorot_uniform', 
                          bias_initializer='zeros', depthwise_regularizer=None, pointwise_regularizer=None, 
                          bias_regularizer=None, activity_regularizer=None, depthwise_constraint=None,
                          pointwise_constraint=None, bias_constraint=None))
model.add(BatchNormalization())
model.add(Activation(elu))
model.add(MaxPooling2D(pool_size=(2, 2), padding="same"))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512, activation= swish ))
model.add(RepeatVector(1))
model.add(SimpleRNN(80, activation= elu, kernel_initializer='glorot_uniform'))
model.add(Dense(512, activation= swish ))
model.add(Dense(6, activation='softmax'))
 
model.compile(optimizer='adagrad', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


model.fit(train_images, train_labels, batch_size=32, validation_split=0.09,callbacks=[early_stop], 
          epochs=200, verbose=1)
model.evaluate(test_images, test_labels, batch_size=32)
## END ###

from keras.utils import plot_model
from google.colab import files

plot_model(model, to_file='model.png')
files.download('model.png')

y_predict = model.predict(x_test)
y_enco = np.argmax(y_predict, axis=1)

from google.colab import files
with open("val_loss: 0.3101 - val_acc: 0.8903_answer.txt", "w") as txt_file:
  for line in y_enco:
    
    txt_file.write(str(int(line))+"\n")

files.download("val_loss: 0.3101 - val_acc: 0.8903_answer.txt")

u, c = np.unique(y_enco, return_counts=True)
dict(zip(u, c))
