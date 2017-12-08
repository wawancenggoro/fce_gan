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
    if dataset=='mnist':
        # load data
        (X_train, y_train), (X_test, y_test) = mnist.load_data()
        #IPython.embed()
        # reshape to be [samples][pixels][width][height]
        X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32')
        X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32')
        #IPython.embed()
        
        # normalize inputs from 0-255 to 0-1
        X_train = X_train / 255
        X_test = X_test / 255
        X_val = X_test
        # one hot encode outputs
        y_train = np_utils.to_categorical(y_train)
        y_test = np_utils.to_categorical(y_test)
        y_val = y_test
        num_classes = y_test.shape[1]
        input_shape=(28, 28, 1)
    elif dataset=='CelebA':
        # load data CelebA
        f=h5py.File('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped.hdf5','r')
        X_train=f['features'][0:1000]
        X_val=f['features'][1000:1100]
        X_test=f['features'][1100:1200]
        y_train=f['targets'][0:1000]
        y_val=f['targets'][1000:1100]
        y_test=f['targets'][1100:1200]
        
        
        # normalize inputs from 0-255 to 0-1
        X_train = X_train / 255
        X_test = X_test / 255
        num_classes = y_test.shape[1]
        input_shape=(218, 178, 3)
    elif dataset=='CelebAHDF5':
        # load data CelebA
        X_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_normalized.hdf5', 'features')
        y_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_normalized.hdf5', 'targets')
#        X_train=X_data[0:162770]
#        X_val=X_data[162770:182637]
#        X_test=X_data[182637:202599]
#        y_train=y_data[0:162770]
#        y_val=y_data[162770:182637]
#        y_test=y_data[182637:202599]
        X_train=X_data
        X_val=X_data
        X_test=X_data
        y_train=y_data
        y_val=y_data
        y_test=y_data
        
        # normalize inputs from 0-255 to 0-1
#        X_train = X_train / 255
#        X_test = X_test / 255
        num_classes = y_test.shape[1]
        input_shape=(218, 178, 3)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes, input_shape
    
def hdf5_generator(dataset,set_type):
    i=0    
    # CelebA data size: train = 162770, valid = 19867, test = 19962
    if dataset=='CelebA':    
        if set_type=='train':
            X_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_train.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_train.hdf5', 'targets')
            size = 162770        
            
        elif set_type=='valid':
            X_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_valid.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_valid.hdf5', 'targets')
            size = 19867        
            
        elif set_type=='test':
            X_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_test.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_test.hdf5', 'targets')
            size = 19962   
            
        if set_type=='train_cls5':
            X_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_train_5cls.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_train_5cls.hdf5', 'targets')
            size = 122077        
            
        elif set_type=='valid_cls5':
            X_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_valid_5cls.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_valid_5cls.hdf5', 'targets')
            size = 15138        
            
        elif set_type=='test_cls5':
            X_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_test_5cls.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_test_5cls.hdf5', 'targets')
            size = 14724          
            
        elif set_type=='all':
            X_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped.hdf5', 'targets')
            size = 202599        
                
        elif set_type=='try':
            X_data = HDF5Matrix('mytestfile.hdf5', 'features')
            y_data = HDF5Matrix('mytestfile.hdf5', 'targets')    
            size = 20        
                    
        while 1:
            X_single = X_data[i%size].reshape((1, 218, 178, 3))
#            y_single = y_data[i%size].reshape((1, 40))
            y_single = y_data[i%size].reshape((1, 5))
#            X_single = X_data[i:i+1]
#            y_single = y_data[i:i+1]
            yield(X_single, y_single)
            i+=1

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

# load data CelebA
#f=h5py.File('../../data/CelebAHDF5/celeba_aligned_cropped.hdf5','r')
#X_train=f['features'][0:1000]
#X_test=f['features'][1000:1100]
#y_train=f['targets'][0:1000]
#y_test=f['targets'][1000:1100]
#
#
## normalize inputs from 0-255 to 0-1
#X_train = X_train / 255
#X_test = X_test / 255
#num_classes = y_test.shape[1]

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

def gan_dis_model_cel5():
    inp = Input(shape=(218,178,3))
    i = inp

    ndf=24
    
    def conv(i,nop,kw,std=1,usebn=True,bm='same',name=''):
        i = Conv2D(nop,kernel_size=(kw,kw),padding=bm,strides=(std,std), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv'+name)(i)
        if usebn:
            i = BatchNormalization(name='bn'+name)(i)
        i = Activation('relu',name='relu'+name)(i)
        return i
    
    # only 5
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
    i0 = conv(i,ndf*8,4,std=2,name='5_cel0')
    i0 = concat_diff(i0)
    i0 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel0')(i0)
    i1 = conv(i,ndf*8,4,std=2,name='5_cel1')
    i1 = concat_diff(i1)
    i1 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel1')(i1)
    i2 = conv(i,ndf*8,4,std=2,name='5_cel2')
    i2 = concat_diff(i2)
    i2 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel2')(i2)
    i3 = conv(i,ndf*8,4,std=2,name='5_cel3')
    i3 = concat_diff(i3)
    i3 = Conv2D(39,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel3')(i3)
    i4 = conv(i,ndf*8,4,std=2,name='5_cel4')
    i4 = concat_diff(i4)
    i4 = Conv2D(39,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel4')(i4)
    i = Concatenate()([i0,i1,i2,i3,i4])
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

def gan_dis_model_cel45():
    inp = Input(shape=(218,178,3))
    i = inp

    ndf=24
    
    def conv(i,nop,kw,std=1,usebn=True,bm='same',name=''):
        i = Conv2D(nop,kernel_size=(kw,kw),padding=bm,strides=(std,std), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv'+name)(i)
        if usebn:
            i = BatchNormalization(name='bn'+name)(i)
        i = Activation('relu',name='relu'+name)(i)
        return i

    # 4 and 5
    i = conv(i,ndf*1,4,std=2,name='0',usebn=False)
    i = concat_diff(i)
    i = conv(i,ndf*2,4,std=2,name='1')
    i = concat_diff(i)
    i = conv(i,ndf*4,4,std=2,name='2')
    i = concat_diff(i)
    i = conv(i,ndf*8,4,std=2,name='3')
    i = concat_diff(i)
    i0 = conv(i,ndf*8,4,std=2,name='4_cel0')
    i0 = concat_diff(i0)
    i1 = conv(i,ndf*8,4,std=2,name='4_cel1')
    i1 = concat_diff(i1)
    i2 = conv(i,ndf*8,4,std=2,name='4_cel2')
    i2 = concat_diff(i2)
    i3 = conv(i,ndf*8,4,std=2,name='4_cel3')
    i3 = concat_diff(i3)
    i4 = conv(i,ndf*8,4,std=2,name='4_cel4')
    i4 = concat_diff(i4)
    i0 = conv(i0,ndf*8,4,std=2,name='5_cel0')
    i0 = concat_diff(i0)
    i0 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel0')(i0)
    i1 = conv(i1,ndf*8,4,std=2,name='5_cel1')
    i1 = concat_diff(i1)
    i1 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel1')(i1)
    i2 = conv(i2,ndf*8,4,std=2,name='5_cel2')
    i2 = concat_diff(i2)
    i2 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel2')(i2)
    i3 = conv(i3,ndf*8,4,std=2,name='5_cel3')
    i3 = concat_diff(i3)
    i3 = Conv2D(39,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel3')(i3)
    i4 = conv(i4,ndf*8,4,std=2,name='5_cel4')
    i4 = concat_diff(i4)
    i4 = Conv2D(39,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel4')(i4)
    i = Concatenate()([i0,i1,i2,i3,i4])
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

def gan_dis_model_cel345():
    inp = Input(shape=(218,178,3))
    i = inp

    ndf=24
    
    def conv(i,nop,kw,std=1,usebn=True,bm='same',name=''):
        i = Conv2D(nop,kernel_size=(kw,kw),padding=bm,strides=(std,std), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv'+name)(i)
        if usebn:
            i = BatchNormalization(name='bn'+name)(i)
        i = Activation('relu',name='relu'+name)(i)
        return i

    # 3, 4, and 5
    i = conv(i,ndf*1,4,std=2,name='0',usebn=False)
    i = concat_diff(i)
    i = conv(i,ndf*2,4,std=2,name='1')
    i = concat_diff(i)
    i = conv(i,ndf*4,4,std=2,name='2')
    i = concat_diff(i)
    i0 = conv(i,ndf*8,4,std=2,name='3_cel0')
    i0 = concat_diff(i0)
    i1 = conv(i,ndf*8,4,std=2,name='3_cel1')
    i1 = concat_diff(i1)
    i2 = conv(i,ndf*8,4,std=2,name='3_cel2')
    i2 = concat_diff(i2)
    i3 = conv(i,ndf*8,4,std=2,name='3_cel3')
    i3 = concat_diff(i3)
    i4 = conv(i,ndf*8,4,std=2,name='3_cel4')
    i4 = concat_diff(i4)
    i0 = conv(i0,ndf*8,4,std=2,name='4_cel0')
    i0 = concat_diff(i0)
    i1 = conv(i1,ndf*8,4,std=2,name='4_cel1')
    i1 = concat_diff(i1)
    i2 = conv(i2,ndf*8,4,std=2,name='4_cel2')
    i2 = concat_diff(i2)
    i3 = conv(i3,ndf*8,4,std=2,name='4_cel3')
    i3 = concat_diff(i3)
    i4 = conv(i4,ndf*8,4,std=2,name='4_cel4')
    i4 = concat_diff(i4)
    i0 = conv(i0,ndf*8,4,std=2,name='5_cel0')
    i0 = concat_diff(i0)
    i0 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel0')(i0)
    i1 = conv(i1,ndf*8,4,std=2,name='5_cel1')
    i1 = concat_diff(i1)
    i1 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel1')(i1)
    i2 = conv(i2,ndf*8,4,std=2,name='5_cel2')
    i2 = concat_diff(i2)
    i2 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel2')(i2)
    i3 = conv(i3,ndf*8,4,std=2,name='5_cel3')
    i3 = concat_diff(i3)
    i3 = Conv2D(39,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel3')(i3)
    i4 = conv(i4,ndf*8,4,std=2,name='5_cel4')
    i4 = concat_diff(i4)
    i4 = Conv2D(39,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel4')(i4)
    i = Concatenate()([i0,i1,i2,i3,i4])
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

def gan_dis_model_cel2345():
    inp = Input(shape=(218,178,3))
    i = inp

    ndf=24
    
    def conv(i,nop,kw,std=1,usebn=True,bm='same',name=''):
        i = Conv2D(nop,kernel_size=(kw,kw),padding=bm,strides=(std,std), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv'+name)(i)
        if usebn:
            i = BatchNormalization(name='bn'+name)(i)
        i = Activation('relu',name='relu'+name)(i)
        return i
    # 2, 3, 4, and 5
    i = conv(i,ndf*1,4,std=2,name='0',usebn=False)
    i = concat_diff(i)
    i = conv(i,ndf*2,4,std=2,name='1')
    i = concat_diff(i)
    i0 = conv(i,ndf*4,4,std=2,name='2_cel0')
    i0 = concat_diff(i0)
    i1 = conv(i,ndf*4,4,std=2,name='2_cel1')
    i1 = concat_diff(i1)
    i2 = conv(i,ndf*4,4,std=2,name='2_cel2')
    i2 = concat_diff(i2)
    i3 = conv(i,ndf*4,4,std=2,name='2_cel3')
    i3 = concat_diff(i3)
    i4 = conv(i,ndf*4,4,std=2,name='2_cel4')
    i4 = concat_diff(i4)
    i0 = conv(i0,ndf*8,4,std=2,name='3_cel0')
    i0 = concat_diff(i0)
    i1 = conv(i1,ndf*8,4,std=2,name='3_cel1')
    i1 = concat_diff(i1)
    i2 = conv(i2,ndf*8,4,std=2,name='3_cel2')
    i2 = concat_diff(i2)
    i3 = conv(i3,ndf*8,4,std=2,name='3_cel3')
    i3 = concat_diff(i3)
    i4 = conv(i4,ndf*8,4,std=2,name='3_cel4')
    i4 = concat_diff(i4)
    i0 = conv(i0,ndf*8,4,std=2,name='4_cel0')
    i0 = concat_diff(i0)
    i1 = conv(i1,ndf*8,4,std=2,name='4_cel1')
    i1 = concat_diff(i1)
    i2 = conv(i2,ndf*8,4,std=2,name='4_cel2')
    i2 = concat_diff(i2)
    i3 = conv(i3,ndf*8,4,std=2,name='4_cel3')
    i3 = concat_diff(i3)
    i4 = conv(i4,ndf*8,4,std=2,name='4_cel4')
    i4 = concat_diff(i4)
    i0 = conv(i0,ndf*8,4,std=2,name='5_cel0')
    i0 = concat_diff(i0)
    i0 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel0')(i0)
    i1 = conv(i1,ndf*8,4,std=2,name='5_cel1')
    i1 = concat_diff(i1)
    i1 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel1')(i1)
    i2 = conv(i2,ndf*8,4,std=2,name='5_cel2')
    i2 = concat_diff(i2)
    i2 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel2')(i2)
    i3 = conv(i3,ndf*8,4,std=2,name='5_cel3')
    i3 = concat_diff(i3)
    i3 = Conv2D(39,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel3')(i3)
    i4 = conv(i4,ndf*8,4,std=2,name='5_cel4')
    i4 = concat_diff(i4)
    i4 = Conv2D(39,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel4')(i4)
    i = Concatenate()([i0,i1,i2,i3,i4])
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
	
def gan_dis_model_cel12345():
    inp = Input(shape=(218,178,3))
    i = inp

    ndf=24
    
    def conv(i,nop,kw,std=1,usebn=True,bm='same',name=''):
        i = Conv2D(nop,kernel_size=(kw,kw),padding=bm,strides=(std,std), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv'+name)(i)
        if usebn:
            i = BatchNormalization(name='bn'+name)(i)
        i = Activation('relu',name='relu'+name)(i)
        return i
    # 2, 3, 4, and 5
    i = conv(i,ndf*1,4,std=2,name='0',usebn=False)
    i = concat_diff(i)
    i0 = conv(i,ndf*2,4,std=2,name='1_cel0')
    i0 = concat_diff(i0)
    i1 = conv(i,ndf*2,4,std=2,name='1_cel1')
    i1 = concat_diff(i1)
    i2 = conv(i,ndf*2,4,std=2,name='1_cel2')
    i2 = concat_diff(i2)
    i3 = conv(i,ndf*2,4,std=2,name='1_cel3')
    i3 = concat_diff(i3)
    i4 = conv(i,ndf*2,4,std=2,name='1_cel4')
    i4 = concat_diff(i4)
    i0 = conv(i0,ndf*4,4,std=2,name='2_cel0')
    i0 = concat_diff(i0)
    i1 = conv(i1,ndf*4,4,std=2,name='2_cel1')
    i1 = concat_diff(i1)
    i2 = conv(i2,ndf*4,4,std=2,name='2_cel2')
    i2 = concat_diff(i2)
    i3 = conv(i3,ndf*4,4,std=2,name='2_cel3')
    i3 = concat_diff(i3)
    i4 = conv(i4,ndf*4,4,std=2,name='2_cel4')
    i4 = concat_diff(i4)
    i0 = conv(i0,ndf*8,4,std=2,name='3_cel0')
    i0 = concat_diff(i0)
    i1 = conv(i1,ndf*8,4,std=2,name='3_cel1')
    i1 = concat_diff(i1)
    i2 = conv(i2,ndf*8,4,std=2,name='3_cel2')
    i2 = concat_diff(i2)
    i3 = conv(i3,ndf*8,4,std=2,name='3_cel3')
    i3 = concat_diff(i3)
    i4 = conv(i4,ndf*8,4,std=2,name='3_cel4')
    i4 = concat_diff(i4)
    i0 = conv(i0,ndf*8,4,std=2,name='4_cel0')
    i0 = concat_diff(i0)
    i1 = conv(i1,ndf*8,4,std=2,name='4_cel1')
    i1 = concat_diff(i1)
    i2 = conv(i2,ndf*8,4,std=2,name='4_cel2')
    i2 = concat_diff(i2)
    i3 = conv(i3,ndf*8,4,std=2,name='4_cel3')
    i3 = concat_diff(i3)
    i4 = conv(i4,ndf*8,4,std=2,name='4_cel4')
    i4 = concat_diff(i4)
    i0 = conv(i0,ndf*8,4,std=2,name='5_cel0')
    i0 = concat_diff(i0)
    i0 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel0')(i0)
    i1 = conv(i1,ndf*8,4,std=2,name='5_cel1')
    i1 = concat_diff(i1)
    i1 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel1')(i1)
    i2 = conv(i2,ndf*8,4,std=2,name='5_cel2')
    i2 = concat_diff(i2)
    i2 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel2')(i2)
    i3 = conv(i3,ndf*8,4,std=2,name='5_cel3')
    i3 = concat_diff(i3)
    i3 = Conv2D(39,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel3')(i3)
    i4 = conv(i4,ndf*8,4,std=2,name='5_cel4')
    i4 = concat_diff(i4)
    i4 = Conv2D(39,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel4')(i4)
    i = Concatenate()([i0,i1,i2,i3,i4])
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

def gan_dis_model_cel012345():
    inp = Input(shape=(218,178,3))
    i = inp

    ndf=24
    
    def conv(i,nop,kw,std=1,usebn=True,bm='same',name=''):
        i = Conv2D(nop,kernel_size=(kw,kw),padding=bm,strides=(std,std), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv'+name)(i)
        if usebn:
            i = BatchNormalization(name='bn'+name)(i)
        i = Activation('relu',name='relu'+name)(i)
        return i
    # 2, 3, 4, and 5
    i0 = conv(i,ndf*1,4,std=2,name='0_cel0',usebn=False)
    i0 = concat_diff(i0)
    i1 = conv(i,ndf*1,4,std=2,name='0_cel1',usebn=False)
    i1 = concat_diff(i1)
    i2 = conv(i,ndf*1,4,std=2,name='0_cel2',usebn=False)
    i2 = concat_diff(i2)
    i3 = conv(i,ndf*1,4,std=2,name='0_cel3',usebn=False)
    i3 = concat_diff(i3)
    i4 = conv(i,ndf*1,4,std=2,name='0_cel4',usebn=False)
    i4 = concat_diff(i4)
    i0 = conv(i0,ndf*2,4,std=2,name='1_cel0')
    i0 = concat_diff(i0)
    i1 = conv(i1,ndf*2,4,std=2,name='1_cel1')
    i1 = concat_diff(i1)
    i2 = conv(i2,ndf*2,4,std=2,name='1_cel2')
    i2 = concat_diff(i2)
    i3 = conv(i3,ndf*2,4,std=2,name='1_cel3')
    i3 = concat_diff(i3)
    i4 = conv(i4,ndf*2,4,std=2,name='1_cel4')
    i4 = concat_diff(i4)
    i0 = conv(i0,ndf*4,4,std=2,name='2_cel0')
    i0 = concat_diff(i0)
    i1 = conv(i1,ndf*4,4,std=2,name='2_cel1')
    i1 = concat_diff(i1)
    i2 = conv(i2,ndf*4,4,std=2,name='2_cel2')
    i2 = concat_diff(i2)
    i3 = conv(i3,ndf*4,4,std=2,name='2_cel3')
    i3 = concat_diff(i3)
    i4 = conv(i4,ndf*4,4,std=2,name='2_cel4')
    i4 = concat_diff(i4)
    i0 = conv(i0,ndf*8,4,std=2,name='3_cel0')
    i0 = concat_diff(i0)
    i1 = conv(i1,ndf*8,4,std=2,name='3_cel1')
    i1 = concat_diff(i1)
    i2 = conv(i2,ndf*8,4,std=2,name='3_cel2')
    i2 = concat_diff(i2)
    i3 = conv(i3,ndf*8,4,std=2,name='3_cel3')
    i3 = concat_diff(i3)
    i4 = conv(i4,ndf*8,4,std=2,name='3_cel4')
    i4 = concat_diff(i4)
    i0 = conv(i0,ndf*8,4,std=2,name='4_cel0')
    i0 = concat_diff(i0)
    i1 = conv(i1,ndf*8,4,std=2,name='4_cel1')
    i1 = concat_diff(i1)
    i2 = conv(i2,ndf*8,4,std=2,name='4_cel2')
    i2 = concat_diff(i2)
    i3 = conv(i3,ndf*8,4,std=2,name='4_cel3')
    i3 = concat_diff(i3)
    i4 = conv(i4,ndf*8,4,std=2,name='4_cel4')
    i4 = concat_diff(i4)
    i0 = conv(i0,ndf*8,4,std=2,name='5_cel0')
    i0 = concat_diff(i0)
    i0 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel0')(i0)
    i1 = conv(i1,ndf*8,4,std=2,name='5_cel1')
    i1 = concat_diff(i1)
    i1 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel1')(i1)
    i2 = conv(i2,ndf*8,4,std=2,name='5_cel2')
    i2 = concat_diff(i2)
    i2 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel2')(i2)
    i3 = conv(i3,ndf*8,4,std=2,name='5_cel3')
    i3 = concat_diff(i3)
    i3 = Conv2D(39,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel3')(i3)
    i4 = conv(i4,ndf*8,4,std=2,name='5_cel4')
    i4 = concat_diff(i4)
    i4 = Conv2D(39,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel4')(i4)
    i = Concatenate()([i0,i1,i2,i3,i4])
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
#X_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped.hdf5', 'features', normalizer=normalize_pixel) 
#IPython.embed()

#(X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes, input_shape = load_data('CelebAHDF5')
num_classes, input_shape = load_data_attr('CelebA_cls5')
# build the model
#model = Xception(include_top=True, weights=None, input_shape=input_shape, classes=num_classes)

#fce-gan
model0 = gan_dis_model_original()
model0.load_weights('/home/wawan/git/fce_gan/save/dm_fce_0.hdf5')

model1 = gan_dis_model_original()
model1.load_weights('/home/wawan/git/fce_gan/save/dm_fce_1.hdf5')

model2 = gan_dis_model_original()
model2.load_weights('/home/wawan/git/fce_gan/save/dm_fce_2.hdf5')

model3 = gan_dis_model_original()
model3.load_weights('/home/wawan/git/fce_gan/save/dm_fce_3.hdf5')

model4 = gan_dis_model_original()
model4.load_weights('/home/wawan/git/fce_gan/save/dm_fce_4.hdf5')

model = gan_dis_model_cel012345()
# # IPython.embed()

# # only 5
# weights_conv5_cel0 = model0.layers[25].get_weights()
# weights_bn5_cel0 = model0.layers[26].get_weights()
# weights_conv5_cel1 = model1.layers[25].get_weights()
# weights_bn5_cel1 = model1.layers[26].get_weights()
# weights_conv5_cel2 = model2.layers[25].get_weights()
# weights_bn5_cel2 = model2.layers[26].get_weights()
# weights_conv5_cel3 = model3.layers[25].get_weights()
# weights_bn5_cel3 = model3.layers[26].get_weights()
# weights_conv5_cel4 = model4.layers[25].get_weights()
# weights_bn5_cel4 = model4.layers[26].get_weights()

# model.layers[25].set_weights(weights_conv5_cel0)
# model.layers[26].set_weights(weights_conv5_cel1)
# model.layers[27].set_weights(weights_conv5_cel2)
# model.layers[28].set_weights(weights_conv5_cel3)
# model.layers[29].set_weights(weights_conv5_cel4)

# model.layers[30].set_weights(weights_bn5_cel0)
# model.layers[31].set_weights(weights_bn5_cel1)
# model.layers[32].set_weights(weights_bn5_cel2)
# model.layers[33].set_weights(weights_bn5_cel3)
# model.layers[34].set_weights(weights_bn5_cel4)

# model.layers[25].trainable = False
# model.layers[26].trainable = False
# model.layers[27].trainable = False
# model.layers[28].trainable = False
# model.layers[29].trainable = False
# model.layers[30].trainable = False
# model.layers[31].trainable = False
# model.layers[32].trainable = False
# model.layers[33].trainable = False
# model.layers[34].trainable = False

# 4 and 5
# weights_conv4_cel0 = model0.layers[20].get_weights()
# weights_bn4_cel0 = model0.layers[21].get_weights()
# weights_conv4_cel1 = model1.layers[20].get_weights()
# weights_bn4_cel1 = model1.layers[21].get_weights()
# weights_conv4_cel2 = model2.layers[20].get_weights()
# weights_bn4_cel2 = model2.layers[21].get_weights()
# weights_conv4_cel3 = model3.layers[20].get_weights()
# weights_bn4_cel3 = model3.layers[21].get_weights()
# weights_conv4_cel4 = model4.layers[20].get_weights()
# weights_bn4_cel4 = model4.layers[21].get_weights()

# model.layers[20].set_weights(weights_conv4_cel0)
# model.layers[21].set_weights(weights_conv4_cel1)
# model.layers[22].set_weights(weights_conv4_cel2)
# model.layers[23].set_weights(weights_conv4_cel3)
# model.layers[24].set_weights(weights_conv4_cel4)
# model.layers[25].set_weights(weights_bn4_cel0)
# model.layers[26].set_weights(weights_bn4_cel1)
# model.layers[27].set_weights(weights_bn4_cel2)
# model.layers[28].set_weights(weights_bn4_cel3)
# model.layers[29].set_weights(weights_bn4_cel4)

# # model.layers[20].trainable = False
# # model.layers[21].trainable = False
# # model.layers[22].trainable = False
# # model.layers[23].trainable = False
# # model.layers[24].trainable = False
# # model.layers[25].trainable = False
# # model.layers[26].trainable = False
# # model.layers[27].trainable = False
# # model.layers[28].trainable = False
# # model.layers[29].trainable = False

# weights_conv5_cel0 = model0.layers[25].get_weights()
# weights_bn5_cel0 = model0.layers[26].get_weights()
# weights_conv5_cel1 = model1.layers[25].get_weights()
# weights_bn5_cel1 = model1.layers[26].get_weights()
# weights_conv5_cel2 = model2.layers[25].get_weights()
# weights_bn5_cel2 = model2.layers[26].get_weights()
# weights_conv5_cel3 = model3.layers[25].get_weights()
# weights_bn5_cel3 = model3.layers[26].get_weights()
# weights_conv5_cel4 = model4.layers[25].get_weights()
# weights_bn5_cel4 = model4.layers[26].get_weights()

# model.layers[45].set_weights(weights_conv5_cel0)
# model.layers[46].set_weights(weights_conv5_cel1)
# model.layers[47].set_weights(weights_conv5_cel2)
# model.layers[48].set_weights(weights_conv5_cel3)
# model.layers[49].set_weights(weights_conv5_cel4)
# model.layers[50].set_weights(weights_bn5_cel0)
# model.layers[51].set_weights(weights_bn5_cel1)
# model.layers[52].set_weights(weights_bn5_cel2)
# model.layers[53].set_weights(weights_bn5_cel3)
# model.layers[54].set_weights(weights_bn5_cel4)

# # model.layers[45].trainable = False
# # model.layers[46].trainable = False
# # model.layers[47].trainable = False
# # model.layers[48].trainable = False
# # model.layers[49].trainable = False
# # model.layers[50].trainable = False
# # model.layers[51].trainable = False
# # model.layers[52].trainable = False
# # model.layers[53].trainable = False
# # model.layers[54].trainable = False


# 3, 4, and 5
# weights_conv3_cel0 = model0.layers[15].get_weights()
# weights_bn3_cel0 = model0.layers[16].get_weights()
# weights_conv3_cel1 = model1.layers[15].get_weights()
# weights_bn3_cel1 = model1.layers[16].get_weights()
# weights_conv3_cel2 = model2.layers[15].get_weights()
# weights_bn3_cel2 = model2.layers[16].get_weights()
# weights_conv3_cel3 = model3.layers[15].get_weights()
# weights_bn3_cel3 = model3.layers[16].get_weights()
# weights_conv3_cel4 = model4.layers[15].get_weights()
# weights_bn3_cel4 = model4.layers[16].get_weights()

# model.layers[15].set_weights(weights_conv3_cel0)
# model.layers[16].set_weights(weights_conv3_cel1)
# model.layers[17].set_weights(weights_conv3_cel2)
# model.layers[18].set_weights(weights_conv3_cel3)
# model.layers[19].set_weights(weights_conv3_cel4)
# model.layers[20].set_weights(weights_bn3_cel0)
# model.layers[21].set_weights(weights_bn3_cel1)
# model.layers[22].set_weights(weights_bn3_cel2)
# model.layers[23].set_weights(weights_bn3_cel3)
# model.layers[24].set_weights(weights_bn3_cel4)

# # model.layers[15].trainable = False
# # model.layers[16].trainable = False
# # model.layers[17].trainable = False
# # model.layers[18].trainable = False
# # model.layers[19].trainable = False
# # model.layers[20].trainable = False
# # model.layers[21].trainable = False
# # model.layers[22].trainable = False
# # model.layers[23].trainable = False
# # model.layers[24].trainable = False

# weights_conv4_cel0 = model0.layers[20].get_weights()
# weights_bn4_cel0 = model0.layers[21].get_weights()
# weights_conv4_cel1 = model1.layers[20].get_weights()
# weights_bn4_cel1 = model1.layers[21].get_weights()
# weights_conv4_cel2 = model2.layers[20].get_weights()
# weights_bn4_cel2 = model2.layers[21].get_weights()
# weights_conv4_cel3 = model3.layers[20].get_weights()
# weights_bn4_cel3 = model3.layers[21].get_weights()
# weights_conv4_cel4 = model4.layers[20].get_weights()
# weights_bn4_cel4 = model4.layers[21].get_weights()

# model.layers[40].set_weights(weights_conv4_cel0)
# model.layers[41].set_weights(weights_conv4_cel1)
# model.layers[42].set_weights(weights_conv4_cel2)
# model.layers[43].set_weights(weights_conv4_cel3)
# model.layers[44].set_weights(weights_conv4_cel4)
# model.layers[45].set_weights(weights_bn4_cel0)
# model.layers[46].set_weights(weights_bn4_cel1)
# model.layers[47].set_weights(weights_bn4_cel2)
# model.layers[48].set_weights(weights_bn4_cel3)
# model.layers[49].set_weights(weights_bn4_cel4)

# # model.layers[40].trainable = False
# # model.layers[41].trainable = False
# # model.layers[42].trainable = False
# # model.layers[43].trainable = False
# # model.layers[44].trainable = False
# # model.layers[45].trainable = False
# # model.layers[46].trainable = False
# # model.layers[47].trainable = False
# # model.layers[48].trainable = False
# # model.layers[49].trainable = False

# weights_conv5_cel0 = model0.layers[25].get_weights()
# weights_bn5_cel0 = model0.layers[26].get_weights()
# weights_conv5_cel1 = model1.layers[25].get_weights()
# weights_bn5_cel1 = model1.layers[26].get_weights()
# weights_conv5_cel2 = model2.layers[25].get_weights()
# weights_bn5_cel2 = model2.layers[26].get_weights()
# weights_conv5_cel3 = model3.layers[25].get_weights()
# weights_bn5_cel3 = model3.layers[26].get_weights()
# weights_conv5_cel4 = model4.layers[25].get_weights()
# weights_bn5_cel4 = model4.layers[26].get_weights()

# model.layers[65].set_weights(weights_conv5_cel0)
# model.layers[66].set_weights(weights_conv5_cel1)
# model.layers[67].set_weights(weights_conv5_cel2)
# model.layers[68].set_weights(weights_conv5_cel3)
# model.layers[69].set_weights(weights_conv5_cel4)
# model.layers[70].set_weights(weights_bn5_cel0)
# model.layers[71].set_weights(weights_bn5_cel1)
# model.layers[72].set_weights(weights_bn5_cel2)
# model.layers[73].set_weights(weights_bn5_cel3)
# model.layers[74].set_weights(weights_bn5_cel4)

# # model.layers[65].trainable = False
# # model.layers[66].trainable = False
# # model.layers[67].trainable = False
# # model.layers[68].trainable = False
# # model.layers[69].trainable = False
# # model.layers[70].trainable = False
# # model.layers[71].trainable = False
# # model.layers[72].trainable = False
# # model.layers[73].trainable = False
# # model.layers[74].trainable = False

# # 2, 3, 4, and 5
# weights_conv2_cel0 = model0.layers[10].get_weights()
# weights_bn2_cel0 = model0.layers[11].get_weights()
# weights_conv2_cel1 = model1.layers[10].get_weights()
# weights_bn2_cel1 = model1.layers[11].get_weights()
# weights_conv2_cel2 = model2.layers[10].get_weights()
# weights_bn2_cel2 = model2.layers[11].get_weights()
# weights_conv2_cel3 = model3.layers[10].get_weights()
# weights_bn2_cel3 = model3.layers[11].get_weights()
# weights_conv2_cel4 = model4.layers[10].get_weights()
# weights_bn2_cel4 = model4.layers[11].get_weights()

# model.layers[10].set_weights(weights_conv2_cel0)
# model.layers[11].set_weights(weights_conv2_cel1)
# model.layers[12].set_weights(weights_conv2_cel2)
# model.layers[13].set_weights(weights_conv2_cel3)
# model.layers[14].set_weights(weights_conv2_cel4)
# model.layers[15].set_weights(weights_bn2_cel0)
# model.layers[16].set_weights(weights_bn2_cel1)
# model.layers[17].set_weights(weights_bn2_cel2)
# model.layers[18].set_weights(weights_bn2_cel3)
# model.layers[19].set_weights(weights_bn2_cel4)

# # model.layers[10].trainable = False
# # model.layers[11].trainable = False
# # model.layers[12].trainable = False
# # model.layers[13].trainable = False
# # model.layers[14].trainable = False
# # model.layers[15].trainable = False
# # model.layers[16].trainable = False
# # model.layers[17].trainable = False
# # model.layers[18].trainable = False
# # model.layers[19].trainable = False

# weights_conv3_cel0 = model0.layers[15].get_weights()
# weights_bn3_cel0 = model0.layers[16].get_weights()
# weights_conv3_cel1 = model1.layers[15].get_weights()
# weights_bn3_cel1 = model1.layers[16].get_weights()
# weights_conv3_cel2 = model2.layers[15].get_weights()
# weights_bn3_cel2 = model2.layers[16].get_weights()
# weights_conv3_cel3 = model3.layers[15].get_weights()
# weights_bn3_cel3 = model3.layers[16].get_weights()
# weights_conv3_cel4 = model4.layers[15].get_weights()
# weights_bn3_cel4 = model4.layers[16].get_weights()

# model.layers[35].set_weights(weights_conv3_cel0)
# model.layers[36].set_weights(weights_conv3_cel1)
# model.layers[37].set_weights(weights_conv3_cel2)
# model.layers[38].set_weights(weights_conv3_cel3)
# model.layers[39].set_weights(weights_conv3_cel4)
# model.layers[40].set_weights(weights_bn3_cel0)
# model.layers[41].set_weights(weights_bn3_cel1)
# model.layers[42].set_weights(weights_bn3_cel2)
# model.layers[43].set_weights(weights_bn3_cel3)
# model.layers[44].set_weights(weights_bn3_cel4)

# # model.layers[35].trainable = False
# # model.layers[36].trainable = False
# # model.layers[37].trainable = False
# # model.layers[38].trainable = False
# # model.layers[39].trainable = False
# # model.layers[40].trainable = False
# # model.layers[41].trainable = False
# # model.layers[42].trainable = False
# # model.layers[43].trainable = False
# # model.layers[44].trainable = False

# weights_conv4_cel0 = model0.layers[20].get_weights()
# weights_bn4_cel0 = model0.layers[21].get_weights()
# weights_conv4_cel1 = model1.layers[20].get_weights()
# weights_bn4_cel1 = model1.layers[21].get_weights()
# weights_conv4_cel2 = model2.layers[20].get_weights()
# weights_bn4_cel2 = model2.layers[21].get_weights()
# weights_conv4_cel3 = model3.layers[20].get_weights()
# weights_bn4_cel3 = model3.layers[21].get_weights()
# weights_conv4_cel4 = model4.layers[20].get_weights()
# weights_bn4_cel4 = model4.layers[21].get_weights()

# model.layers[60].set_weights(weights_conv4_cel0)
# model.layers[61].set_weights(weights_conv4_cel1)
# model.layers[62].set_weights(weights_conv4_cel2)
# model.layers[63].set_weights(weights_conv4_cel3)
# model.layers[64].set_weights(weights_conv4_cel4)
# model.layers[65].set_weights(weights_bn4_cel0)
# model.layers[66].set_weights(weights_bn4_cel1)
# model.layers[67].set_weights(weights_bn4_cel2)
# model.layers[68].set_weights(weights_bn4_cel3)
# model.layers[69].set_weights(weights_bn4_cel4)

# # model.layers[60].trainable = False
# # model.layers[61].trainable = False
# # model.layers[62].trainable = False
# # model.layers[63].trainable = False
# # model.layers[64].trainable = False
# # model.layers[65].trainable = False
# # model.layers[66].trainable = False
# # model.layers[67].trainable = False
# # model.layers[68].trainable = False
# # model.layers[69].trainable = False

# weights_conv5_cel0 = model0.layers[25].get_weights()
# weights_bn5_cel0 = model0.layers[26].get_weights()
# weights_conv5_cel1 = model1.layers[25].get_weights()
# weights_bn5_cel1 = model1.layers[26].get_weights()
# weights_conv5_cel2 = model2.layers[25].get_weights()
# weights_bn5_cel2 = model2.layers[26].get_weights()
# weights_conv5_cel3 = model3.layers[25].get_weights()
# weights_bn5_cel3 = model3.layers[26].get_weights()
# weights_conv5_cel4 = model4.layers[25].get_weights()
# weights_bn5_cel4 = model4.layers[26].get_weights()

# model.layers[85].set_weights(weights_conv5_cel0)
# model.layers[86].set_weights(weights_conv5_cel1)
# model.layers[87].set_weights(weights_conv5_cel2)
# model.layers[88].set_weights(weights_conv5_cel3)
# model.layers[89].set_weights(weights_conv5_cel4)
# model.layers[90].set_weights(weights_bn5_cel0)
# model.layers[91].set_weights(weights_bn5_cel1)
# model.layers[92].set_weights(weights_bn5_cel2)
# model.layers[93].set_weights(weights_bn5_cel3)
# model.layers[94].set_weights(weights_bn5_cel4)

# # model.layers[85].trainable = False
# # model.layers[86].trainable = False
# # model.layers[87].trainable = False
# # model.layers[88].trainable = False
# # model.layers[89].trainable = False
# # model.layers[90].trainable = False
# # model.layers[91].trainable = False
# # model.layers[92].trainable = False
# # model.layers[93].trainable = False
# # model.layers[94].trainable = False

# weights_conv2_cel0 = model0.layers[10].get_weights()
# weights_bn2_cel0 = model0.layers[11].get_weights()
# weights_conv2_cel1 = model1.layers[10].get_weights()
# weights_bn2_cel1 = model1.layers[11].get_weights()
# weights_conv2_cel2 = model2.layers[10].get_weights()
# weights_bn2_cel2 = model2.layers[11].get_weights()
# weights_conv2_cel3 = model3.layers[10].get_weights()
# weights_bn2_cel3 = model3.layers[11].get_weights()
# weights_conv2_cel4 = model4.layers[10].get_weights()
# weights_bn2_cel4 = model4.layers[11].get_weights()

# model.layers[10].set_weights(weights_conv2_cel0)
# model.layers[11].set_weights(weights_conv2_cel1)
# model.layers[12].set_weights(weights_conv2_cel2)
# model.layers[13].set_weights(weights_conv2_cel3)
# model.layers[14].set_weights(weights_conv2_cel4)
# model.layers[15].set_weights(weights_bn2_cel0)
# model.layers[16].set_weights(weights_bn2_cel1)
# model.layers[17].set_weights(weights_bn2_cel2)
# model.layers[18].set_weights(weights_bn2_cel3)
# model.layers[19].set_weights(weights_bn2_cel4)

# # model.layers[10].trainable = False
# # model.layers[11].trainable = False
# # model.layers[12].trainable = False
# # model.layers[13].trainable = False
# # model.layers[14].trainable = False
# # model.layers[15].trainable = False
# # model.layers[16].trainable = False
# # model.layers[17].trainable = False
# # model.layers[18].trainable = False
# # model.layers[19].trainable = False

# weights_conv3_cel0 = model0.layers[15].get_weights()
# weights_bn3_cel0 = model0.layers[16].get_weights()
# weights_conv3_cel1 = model1.layers[15].get_weights()
# weights_bn3_cel1 = model1.layers[16].get_weights()
# weights_conv3_cel2 = model2.layers[15].get_weights()
# weights_bn3_cel2 = model2.layers[16].get_weights()
# weights_conv3_cel3 = model3.layers[15].get_weights()
# weights_bn3_cel3 = model3.layers[16].get_weights()
# weights_conv3_cel4 = model4.layers[15].get_weights()
# weights_bn3_cel4 = model4.layers[16].get_weights()

# model.layers[35].set_weights(weights_conv3_cel0)
# model.layers[36].set_weights(weights_conv3_cel1)
# model.layers[37].set_weights(weights_conv3_cel2)
# model.layers[38].set_weights(weights_conv3_cel3)
# model.layers[39].set_weights(weights_conv3_cel4)
# model.layers[40].set_weights(weights_bn3_cel0)
# model.layers[41].set_weights(weights_bn3_cel1)
# model.layers[42].set_weights(weights_bn3_cel2)
# model.layers[43].set_weights(weights_bn3_cel3)
# model.layers[44].set_weights(weights_bn3_cel4)

# # model.layers[35].trainable = False
# # model.layers[36].trainable = False
# # model.layers[37].trainable = False
# # model.layers[38].trainable = False
# # model.layers[39].trainable = False
# # model.layers[40].trainable = False
# # model.layers[41].trainable = False
# # model.layers[42].trainable = False
# # model.layers[43].trainable = False
# # model.layers[44].trainable = False

# weights_conv4_cel0 = model0.layers[20].get_weights()
# weights_bn4_cel0 = model0.layers[21].get_weights()
# weights_conv4_cel1 = model1.layers[20].get_weights()
# weights_bn4_cel1 = model1.layers[21].get_weights()
# weights_conv4_cel2 = model2.layers[20].get_weights()
# weights_bn4_cel2 = model2.layers[21].get_weights()
# weights_conv4_cel3 = model3.layers[20].get_weights()
# weights_bn4_cel3 = model3.layers[21].get_weights()
# weights_conv4_cel4 = model4.layers[20].get_weights()
# weights_bn4_cel4 = model4.layers[21].get_weights()

# model.layers[60].set_weights(weights_conv4_cel0)
# model.layers[61].set_weights(weights_conv4_cel1)
# model.layers[62].set_weights(weights_conv4_cel2)
# model.layers[63].set_weights(weights_conv4_cel3)
# model.layers[64].set_weights(weights_conv4_cel4)
# model.layers[65].set_weights(weights_bn4_cel0)
# model.layers[66].set_weights(weights_bn4_cel1)
# model.layers[67].set_weights(weights_bn4_cel2)
# model.layers[68].set_weights(weights_bn4_cel3)
# model.layers[69].set_weights(weights_bn4_cel4)

# # model.layers[60].trainable = False
# # model.layers[61].trainable = False
# # model.layers[62].trainable = False
# # model.layers[63].trainable = False
# # model.layers[64].trainable = False
# # model.layers[65].trainable = False
# # model.layers[66].trainable = False
# # model.layers[67].trainable = False
# # model.layers[68].trainable = False
# # model.layers[69].trainable = False

# weights_conv5_cel0 = model0.layers[25].get_weights()
# weights_bn5_cel0 = model0.layers[26].get_weights()
# weights_conv5_cel1 = model1.layers[25].get_weights()
# weights_bn5_cel1 = model1.layers[26].get_weights()
# weights_conv5_cel2 = model2.layers[25].get_weights()
# weights_bn5_cel2 = model2.layers[26].get_weights()
# weights_conv5_cel3 = model3.layers[25].get_weights()
# weights_bn5_cel3 = model3.layers[26].get_weights()
# weights_conv5_cel4 = model4.layers[25].get_weights()
# weights_bn5_cel4 = model4.layers[26].get_weights()

# model.layers[85].set_weights(weights_conv5_cel0)
# model.layers[86].set_weights(weights_conv5_cel1)
# model.layers[87].set_weights(weights_conv5_cel2)
# model.layers[88].set_weights(weights_conv5_cel3)
# model.layers[89].set_weights(weights_conv5_cel4)
# model.layers[90].set_weights(weights_bn5_cel0)
# model.layers[91].set_weights(weights_bn5_cel1)
# model.layers[92].set_weights(weights_bn5_cel2)
# model.layers[93].set_weights(weights_bn5_cel3)
# model.layers[94].set_weights(weights_bn5_cel4)

# # model.layers[85].trainable = False
# # model.layers[86].trainable = False
# # model.layers[87].trainable = False
# # model.layers[88].trainable = False
# # model.layers[89].trainable = False
# # model.layers[90].trainable = False
# # model.layers[91].trainable = False
# # model.layers[92].trainable = False
# # model.layers[93].trainable = False
# # model.layers[94].trainable = False


# # 1, 2, 3, 4, and 5
# weights_conv1_cel0 = model0.layers[5].get_weights()
# weights_bn1_cel0 = model0.layers[6].get_weights()
# weights_conv1_cel1 = model1.layers[5].get_weights()
# weights_bn1_cel1 = model1.layers[6].get_weights()
# weights_conv1_cel2 = model2.layers[5].get_weights()
# weights_bn1_cel2 = model2.layers[6].get_weights()
# weights_conv1_cel3 = model3.layers[5].get_weights()
# weights_bn1_cel3 = model3.layers[6].get_weights()
# weights_conv1_cel4 = model4.layers[5].get_weights()
# weights_bn1_cel4 = model4.layers[6].get_weights()

# model.layers[5].set_weights(weights_conv1_cel0)
# model.layers[6].set_weights(weights_conv1_cel1)
# model.layers[7].set_weights(weights_conv1_cel2)
# model.layers[8].set_weights(weights_conv1_cel3)
# model.layers[9].set_weights(weights_conv1_cel4)
# model.layers[10].set_weights(weights_bn1_cel0)
# model.layers[11].set_weights(weights_bn1_cel1)
# model.layers[12].set_weights(weights_bn1_cel2)
# model.layers[13].set_weights(weights_bn1_cel3)
# model.layers[14].set_weights(weights_bn1_cel4)

# # model.layers[5].trainable = False
# # model.layers[6].trainable = False
# # model.layers[7].trainable = False
# # model.layers[8].trainable = False
# # model.layers[9].trainable = False
# # model.layers[10].trainable = False
# # model.layers[11].trainable = False
# # model.layers[12].trainable = False
# # model.layers[13].trainable = False
# # model.layers[14].trainable = False

# weights_conv2_cel0 = model0.layers[10].get_weights()
# weights_bn2_cel0 = model0.layers[11].get_weights()
# weights_conv2_cel1 = model1.layers[10].get_weights()
# weights_bn2_cel1 = model1.layers[11].get_weights()
# weights_conv2_cel2 = model2.layers[10].get_weights()
# weights_bn2_cel2 = model2.layers[11].get_weights()
# weights_conv2_cel3 = model3.layers[10].get_weights()
# weights_bn2_cel3 = model3.layers[11].get_weights()
# weights_conv2_cel4 = model4.layers[10].get_weights()
# weights_bn2_cel4 = model4.layers[11].get_weights()

# model.layers[30].set_weights(weights_conv2_cel0)
# model.layers[31].set_weights(weights_conv2_cel1)
# model.layers[32].set_weights(weights_conv2_cel2)
# model.layers[33].set_weights(weights_conv2_cel3)
# model.layers[34].set_weights(weights_conv2_cel4)
# model.layers[35].set_weights(weights_bn2_cel0)
# model.layers[36].set_weights(weights_bn2_cel1)
# model.layers[37].set_weights(weights_bn2_cel2)
# model.layers[38].set_weights(weights_bn2_cel3)
# model.layers[39].set_weights(weights_bn2_cel4)

# # model.layers[30].trainable = False
# # model.layers[31].trainable = False
# # model.layers[32].trainable = False
# # model.layers[33].trainable = False
# # model.layers[34].trainable = False
# # model.layers[35].trainable = False
# # model.layers[36].trainable = False
# # model.layers[37].trainable = False
# # model.layers[38].trainable = False
# # model.layers[39].trainable = False

# weights_conv3_cel0 = model0.layers[15].get_weights()
# weights_bn3_cel0 = model0.layers[16].get_weights()
# weights_conv3_cel1 = model1.layers[15].get_weights()
# weights_bn3_cel1 = model1.layers[16].get_weights()
# weights_conv3_cel2 = model2.layers[15].get_weights()
# weights_bn3_cel2 = model2.layers[16].get_weights()
# weights_conv3_cel3 = model3.layers[15].get_weights()
# weights_bn3_cel3 = model3.layers[16].get_weights()
# weights_conv3_cel4 = model4.layers[15].get_weights()
# weights_bn3_cel4 = model4.layers[16].get_weights()

# model.layers[55].set_weights(weights_conv3_cel0)
# model.layers[56].set_weights(weights_conv3_cel1)
# model.layers[57].set_weights(weights_conv3_cel2)
# model.layers[58].set_weights(weights_conv3_cel3)
# model.layers[59].set_weights(weights_conv3_cel4)
# model.layers[60].set_weights(weights_bn3_cel0)
# model.layers[61].set_weights(weights_bn3_cel1)
# model.layers[62].set_weights(weights_bn3_cel2)
# model.layers[63].set_weights(weights_bn3_cel3)
# model.layers[64].set_weights(weights_bn3_cel4)

# # model.layers[55].trainable = False
# # model.layers[56].trainable = False
# # model.layers[57].trainable = False
# # model.layers[58].trainable = False
# # model.layers[59].trainable = False
# # model.layers[60].trainable = False
# # model.layers[61].trainable = False
# # model.layers[62].trainable = False
# # model.layers[63].trainable = False
# # model.layers[64].trainable = False

# weights_conv4_cel0 = model0.layers[20].get_weights()
# weights_bn4_cel0 = model0.layers[21].get_weights()
# weights_conv4_cel1 = model1.layers[20].get_weights()
# weights_bn4_cel1 = model1.layers[21].get_weights()
# weights_conv4_cel2 = model2.layers[20].get_weights()
# weights_bn4_cel2 = model2.layers[21].get_weights()
# weights_conv4_cel3 = model3.layers[20].get_weights()
# weights_bn4_cel3 = model3.layers[21].get_weights()
# weights_conv4_cel4 = model4.layers[20].get_weights()
# weights_bn4_cel4 = model4.layers[21].get_weights()

# model.layers[80].set_weights(weights_conv4_cel0)
# model.layers[81].set_weights(weights_conv4_cel1)
# model.layers[82].set_weights(weights_conv4_cel2)
# model.layers[83].set_weights(weights_conv4_cel3)
# model.layers[84].set_weights(weights_conv4_cel4)
# model.layers[85].set_weights(weights_bn4_cel0)
# model.layers[86].set_weights(weights_bn4_cel1)
# model.layers[87].set_weights(weights_bn4_cel2)
# model.layers[88].set_weights(weights_bn4_cel3)
# model.layers[89].set_weights(weights_bn4_cel4)

# # model.layers[80].trainable = False
# # model.layers[81].trainable = False
# # model.layers[82].trainable = False
# # model.layers[83].trainable = False
# # model.layers[84].trainable = False
# # model.layers[85].trainable = False
# # model.layers[86].trainable = False
# # model.layers[87].trainable = False
# # model.layers[88].trainable = False
# # model.layers[89].trainable = False

# weights_conv5_cel0 = model0.layers[25].get_weights()
# weights_bn5_cel0 = model0.layers[26].get_weights()
# weights_conv5_cel1 = model1.layers[25].get_weights()
# weights_bn5_cel1 = model1.layers[26].get_weights()
# weights_conv5_cel2 = model2.layers[25].get_weights()
# weights_bn5_cel2 = model2.layers[26].get_weights()
# weights_conv5_cel3 = model3.layers[25].get_weights()
# weights_bn5_cel3 = model3.layers[26].get_weights()
# weights_conv5_cel4 = model4.layers[25].get_weights()
# weights_bn5_cel4 = model4.layers[26].get_weights()

# model.layers[105].set_weights(weights_conv5_cel0)
# model.layers[106].set_weights(weights_conv5_cel1)
# model.layers[107].set_weights(weights_conv5_cel2)
# model.layers[108].set_weights(weights_conv5_cel3)
# model.layers[109].set_weights(weights_conv5_cel4)
# model.layers[110].set_weights(weights_bn5_cel0)
# model.layers[111].set_weights(weights_bn5_cel1)
# model.layers[112].set_weights(weights_bn5_cel2)
# model.layers[113].set_weights(weights_bn5_cel3)
# model.layers[114].set_weights(weights_bn5_cel4)

# # model.layers[105].trainable = False
# # model.layers[106].trainable = False
# # model.layers[107].trainable = False
# # model.layers[108].trainable = False
# # model.layers[109].trainable = False
# # model.layers[110].trainable = False
# # model.layers[111].trainable = False
# # model.layers[112].trainable = False
# # model.layers[113].trainable = False
# # model.layers[114].trainable = False


# # 0, 1, 2, 3, 4, and 5
weights_conv1_cel0 = model0.layers[1].get_weights()
weights_conv1_cel1 = model1.layers[1].get_weights()
weights_conv1_cel2 = model2.layers[1].get_weights()
weights_conv1_cel3 = model3.layers[1].get_weights()
weights_conv1_cel4 = model4.layers[1].get_weights()

model.layers[1].set_weights(weights_conv1_cel0)
model.layers[2].set_weights(weights_conv1_cel1)
model.layers[3].set_weights(weights_conv1_cel2)
model.layers[4].set_weights(weights_conv1_cel3)
model.layers[5].set_weights(weights_conv1_cel4)

# model.layers[1].trainable = False
# model.layers[2].trainable = False
# model.layers[3].trainable = False
# model.layers[4].trainable = False
# model.layers[5].trainable = False

weights_conv1_cel0 = model0.layers[5].get_weights()
weights_bn1_cel0 = model0.layers[6].get_weights()
weights_conv1_cel1 = model1.layers[5].get_weights()
weights_bn1_cel1 = model1.layers[6].get_weights()
weights_conv1_cel2 = model2.layers[5].get_weights()
weights_bn1_cel2 = model2.layers[6].get_weights()
weights_conv1_cel3 = model3.layers[5].get_weights()
weights_bn1_cel3 = model3.layers[6].get_weights()
weights_conv1_cel4 = model4.layers[5].get_weights()
weights_bn1_cel4 = model4.layers[6].get_weights()

model.layers[21].set_weights(weights_conv1_cel0)
model.layers[22].set_weights(weights_conv1_cel1)
model.layers[23].set_weights(weights_conv1_cel2)
model.layers[24].set_weights(weights_conv1_cel3)
model.layers[25].set_weights(weights_conv1_cel4)
model.layers[26].set_weights(weights_bn1_cel0)
model.layers[27].set_weights(weights_bn1_cel1)
model.layers[28].set_weights(weights_bn1_cel2)
model.layers[29].set_weights(weights_bn1_cel3)
model.layers[30].set_weights(weights_bn1_cel4)

# model.layers[21].trainable = False
# model.layers[22].trainable = False
# model.layers[23].trainable = False
# model.layers[24].trainable = False
# model.layers[25].trainable = False
# model.layers[26].trainable = False
# model.layers[27].trainable = False
# model.layers[28].trainable = False
# model.layers[29].trainable = False
# model.layers[30].trainable = False

weights_conv2_cel0 = model0.layers[10].get_weights()
weights_bn2_cel0 = model0.layers[11].get_weights()
weights_conv2_cel1 = model1.layers[10].get_weights()
weights_bn2_cel1 = model1.layers[11].get_weights()
weights_conv2_cel2 = model2.layers[10].get_weights()
weights_bn2_cel2 = model2.layers[11].get_weights()
weights_conv2_cel3 = model3.layers[10].get_weights()
weights_bn2_cel3 = model3.layers[11].get_weights()
weights_conv2_cel4 = model4.layers[10].get_weights()
weights_bn2_cel4 = model4.layers[11].get_weights()

model.layers[46].set_weights(weights_conv2_cel0)
model.layers[47].set_weights(weights_conv2_cel1)
model.layers[48].set_weights(weights_conv2_cel2)
model.layers[49].set_weights(weights_conv2_cel3)
model.layers[50].set_weights(weights_conv2_cel4)
model.layers[51].set_weights(weights_bn2_cel0)
model.layers[52].set_weights(weights_bn2_cel1)
model.layers[53].set_weights(weights_bn2_cel2)
model.layers[54].set_weights(weights_bn2_cel3)
model.layers[55].set_weights(weights_bn2_cel4)

# model.layers[46].trainable = False
# model.layers[47].trainable = False
# model.layers[48].trainable = False
# model.layers[49].trainable = False
# model.layers[50].trainable = False
# model.layers[51].trainable = False
# model.layers[52].trainable = False
# model.layers[53].trainable = False
# model.layers[54].trainable = False
# model.layers[55].trainable = False

weights_conv3_cel0 = model0.layers[15].get_weights()
weights_bn3_cel0 = model0.layers[16].get_weights()
weights_conv3_cel1 = model1.layers[15].get_weights()
weights_bn3_cel1 = model1.layers[16].get_weights()
weights_conv3_cel2 = model2.layers[15].get_weights()
weights_bn3_cel2 = model2.layers[16].get_weights()
weights_conv3_cel3 = model3.layers[15].get_weights()
weights_bn3_cel3 = model3.layers[16].get_weights()
weights_conv3_cel4 = model4.layers[15].get_weights()
weights_bn3_cel4 = model4.layers[16].get_weights()

model.layers[71].set_weights(weights_conv3_cel0)
model.layers[72].set_weights(weights_conv3_cel1)
model.layers[73].set_weights(weights_conv3_cel2)
model.layers[74].set_weights(weights_conv3_cel3)
model.layers[75].set_weights(weights_conv3_cel4)
model.layers[76].set_weights(weights_bn3_cel0)
model.layers[77].set_weights(weights_bn3_cel1)
model.layers[78].set_weights(weights_bn3_cel2)
model.layers[79].set_weights(weights_bn3_cel3)
model.layers[80].set_weights(weights_bn3_cel4)

# model.layers[71].trainable = False
# model.layers[72].trainable = False
# model.layers[73].trainable = False
# model.layers[74].trainable = False
# model.layers[75].trainable = False
# model.layers[76].trainable = False
# model.layers[77].trainable = False
# model.layers[78].trainable = False
# model.layers[79].trainable = False
# model.layers[80].trainable = False

weights_conv4_cel0 = model0.layers[20].get_weights()
weights_bn4_cel0 = model0.layers[21].get_weights()
weights_conv4_cel1 = model1.layers[20].get_weights()
weights_bn4_cel1 = model1.layers[21].get_weights()
weights_conv4_cel2 = model2.layers[20].get_weights()
weights_bn4_cel2 = model2.layers[21].get_weights()
weights_conv4_cel3 = model3.layers[20].get_weights()
weights_bn4_cel3 = model3.layers[21].get_weights()
weights_conv4_cel4 = model4.layers[20].get_weights()
weights_bn4_cel4 = model4.layers[21].get_weights()

model.layers[96].set_weights(weights_conv4_cel0)
model.layers[97].set_weights(weights_conv4_cel1)
model.layers[98].set_weights(weights_conv4_cel2)
model.layers[99].set_weights(weights_conv4_cel3)
model.layers[100].set_weights(weights_conv4_cel4)
model.layers[101].set_weights(weights_bn4_cel0)
model.layers[102].set_weights(weights_bn4_cel1)
model.layers[103].set_weights(weights_bn4_cel2)
model.layers[104].set_weights(weights_bn4_cel3)
model.layers[105].set_weights(weights_bn4_cel4)

# model.layers[96].trainable = False
# model.layers[97].trainable = False
# model.layers[98].trainable = False
# model.layers[99].trainable = False
# model.layers[100].trainable = False
# model.layers[101].trainable = False
# model.layers[102].trainable = False
# model.layers[103].trainable = False
# model.layers[104].trainable = False
# model.layers[105].trainable = False

weights_conv5_cel0 = model0.layers[25].get_weights()
weights_bn5_cel0 = model0.layers[26].get_weights()
weights_conv5_cel1 = model1.layers[25].get_weights()
weights_bn5_cel1 = model1.layers[26].get_weights()
weights_conv5_cel2 = model2.layers[25].get_weights()
weights_bn5_cel2 = model2.layers[26].get_weights()
weights_conv5_cel3 = model3.layers[25].get_weights()
weights_bn5_cel3 = model3.layers[26].get_weights()
weights_conv5_cel4 = model4.layers[25].get_weights()
weights_bn5_cel4 = model4.layers[26].get_weights()

model.layers[121].set_weights(weights_conv5_cel0)
model.layers[122].set_weights(weights_conv5_cel1)
model.layers[123].set_weights(weights_conv5_cel2)
model.layers[124].set_weights(weights_conv5_cel3)
model.layers[125].set_weights(weights_conv5_cel4)
model.layers[126].set_weights(weights_bn5_cel0)
model.layers[127].set_weights(weights_bn5_cel1)
model.layers[128].set_weights(weights_bn5_cel2)
model.layers[129].set_weights(weights_bn5_cel3)
model.layers[130].set_weights(weights_bn5_cel4)

# model.layers[121].trainable = False
# model.layers[122].trainable = False
# model.layers[123].trainable = False
# model.layers[124].trainable = False
# model.layers[125].trainable = False
# model.layers[126].trainable = False
# model.layers[127].trainable = False
# model.layers[128].trainable = False
# model.layers[129].trainable = False
# model.layers[130].trainable = False


# model = gan_dis_model()
#model = baseline_model_api(input_shape)
model.summary()
filepath="cnn_weights_best.hdf5"
# model=load_model(filepath)
# Compile model
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
#opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.97)
#opt = optimizers.SGD(lr=0.045, momentum=0.9, decay=0.9)
model.compile(optimizer=opt,
          loss=losses.binary_crossentropy,
          metrics=['accuracy'])

def schedule(epoch):
    if epoch<100:
        return epoch*0.001/100
    else:
        return 0.001

          
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=100, verbose=1, mode='auto')
csv_logger = CSVLogger('training.log', append=False)
tensorboard = TensorBoard(log_dir='./tf-logs')
lr_schedule = LearningRateScheduler(schedule)
callbacks_list = [checkpoint, csv_logger, tensorboard]

#IPython.embed()
## Fit the model
#model.fit(X_train[0:162770], y_train[0:162770], validation_data=(X_val[162770:182637], y_val[162770:182637]), epochs=2, batch_size=200, verbose=2)

## Final evaluation of the model
#scores = model.evaluate(X_test[182637:202599], y_test[182637:202599], verbose=0)
#print("Baseline Error: %.2f%%" % (100-scores[1]*100))
#prediction=model.predict(X_test[202599])

# Fit the model with generator
# CelebA data size: train = 162770, valid = 19867, test = 19962
# CelebA cls5 data size: train = 122077, valid = 15138, test = 14724
# callbacks = [TensorBoard(log_dir='./tf-logs')]
batch_size=256
result = model.fit_generator(hdf5_generator('CelebA','train_cls5'),122077//batch_size,1000,initial_epoch=0,validation_data=hdf5_generator('CelebA','valid_cls5'),validation_steps=15138//batch_size,callbacks=callbacks_list)

# Final evaluation of the model with generator
scores = model.evaluate_generator(hdf5_generator('CelebA','test_cls5'),74)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
# prediction=model.predict_generator(hdf5_generator('CelebA','test_cls5'), 74)

IPython.embed()