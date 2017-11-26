# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 15:53:33 2017

@author: wawan
"""

from __future__ import print_function
from __future__ import absolute_import

import warnings

from keras.layers import Concatenate, Maximum, Average
from keras.layers import Lambda

from keras.models import Model
from keras import layers
from keras.layers import Dense
from keras.layers import Input
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import SeparableConv2D
from keras.layers import MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import GlobalMaxPooling2D
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler
from keras.models import load_model


TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'

import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils
from keras import optimizers
import IPython
import h5py
from keras.utils.io_utils import HDF5Matrix
from tensorflow.python import debug as tf_debug
from keras import metrics, losses, regularizers

#sess = K.get_session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#K.set_session(sess)
reg_val = 5e-6

def normalize_pixel(data):
    return data/255-.5

def load_data(dataset):
    if dataset=='CelebAHDF5_cls5_val':
        # load data CelebA
        X_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_valid_5cls.hdf5', 'features', normalizer=normalize_pixel)
        y_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_valid_5cls.hdf5', 'targets')
        
        num_classes = y_test.shape[1]
        input_shape=(218, 178, 3)
    return (X_data, y_data), num_classes, input_shape

def load_data_attr(dataset):
    if dataset=='mnist':
        num_classes = 10
        input_shape=(28, 28, 1)
    elif dataset=='CelebA':   
        num_classes = 40
        input_shape=(218, 178, 3)
    elif dataset=='CelebA_cls5':   
        num_classes = 5
        input_shape=(218, 178, 3)
        
    return num_classes, input_shape

def load_bootstrap():
    return (X_data, y_data)

def concat_diff(i): # batch discrimination -  increase generation diversity.
    # return i
    bv = Lambda(lambda i:K.mean(K.abs(i[:] - K.mean(i,axis=0)),axis=-1,keepdims=True))(i)
    i = Concatenate()([i,bv])
    return i

def gan_dis_model_original():
    inp = Input(shape=(218,178,3))
    i = inp

    ndf=24
    
    def conv(i,nop,kw,std=1,usebn=True,bm='same'):
        i = Conv2D(nop,kernel_size=(kw,kw),padding=bm,strides=(std,std), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val))(i)
        if usebn:
            i = BatchNormalization()(i)
        i = Activation('relu')(i)
        return i
        
    i = conv(i,ndf*1,4,std=2,usebn=False)
    i = concat_diff(i)
    i = conv(i,ndf*2,4,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*4,4,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*8,4,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*8,4,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*8,4,std=2)
    i = concat_diff(i)
    
#    # 1x1
    i = Conv2D(1,(2,2),padding='valid')(i)

    i = Activation('linear',name='conv_exit')(i)
    i = Activation('sigmoid')(i)

#    i = Reshape((1,))(i)

    m = Model(input=inp,output=i)
    return m

def gan_dis_model():
    inp = Input(shape=(218,178,3))
    i = inp

    ndf=24
    
    def conv(i,nop,kw,std=1,usebn=True,bm='same',name=''):
        i = Conv2D(nop,kernel_size=(kw,kw),padding=bm,strides=(std,std), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv'+name)(i)
        if usebn:
            i = BatchNormalization(name='bn'+name)(i)
        i = Activation('relu',name='relu'+name)(i)
        return i
        
    i = conv(i,ndf*1,4,std=2,name='0',usebn=False)
    i = concat_diff(i)
    i = conv(i,ndf*2,4,std=2,name='1')
    i = concat_diff(i)
    i = conv(i,ndf*4,4,std=2,name='2')
    i = concat_diff(i)
    i = conv(i,ndf*8,4,std=2,name='3')
    i = concat_diff(i)
    i = conv(i,ndf*8,4,std=2,name='4')
    i = concat_diff(i)
    i = conv(i,ndf*8,4,std=2,name='5')
    i = concat_diff(i)
    i = Flatten()(i)
    i = Dense(200, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='dense1')(i)
    i = Activation('relu',name='relu_dens1')(i)
    # i = Dense(200, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='dense2')(i)
    # i = Activation('relu',name='relu_dens2')(i)
    i = Dense(num_classes, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='last_dense')(i)
    predictions=Activation('sigmoid')(i)
    model = Model(inputs=inp, outputs=predictions)  
    
    return model
	
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

(X_data, y_data), num_classes, input_shape = load_data('CelebAHDF5_cls5_valid')
# num_classes, input_shape = load_data_attr('CelebA_cls5')

# build the model
model = gan_dis_model()
model.summary()
filepath="cnn_weights_best.hdf5"
# model=load_model(filepath)
# Compile model
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
model.compile(optimizer=opt,
          loss=losses.binary_crossentropy,
          metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')
csv_logger = CSVLogger('training.log', append=False)
tensorboard = TensorBoard(log_dir='./tf-logs')
lr_schedule = LearningRateScheduler(schedule)
callbacks_list = [checkpoint, csv_logger, tensorboard]

batch_size=256
for i in range(1000):
    model=load_model(filepath)
    (X_train, y_train)=load_boostrap()
    result=model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=1, initial_epoch=i, batch_size=batch_size, verbose=2, callbacks=callbacks_list)
    
# Final evaluation of the model with generator
scores = model.evaluate_generator(hdf5_generator('CelebA','test_cls5'),74)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
# prediction=model.predict_generator(hdf5_generator('CelebA','test_cls5'), 74)


# IPython.embed()