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
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger


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
reg_val = 1e-5

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
        f=h5py.File('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped.hdf5','r')
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
        X_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_normalized.hdf5', 'features')
        y_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_normalized.hdf5', 'targets')
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
            X_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_train.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_train.hdf5', 'targets')
            size = 162770        
            
        elif set_type=='valid':
            X_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_valid.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_valid.hdf5', 'targets')
            size = 19867        
            
        elif set_type=='test':
            X_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_test.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_test.hdf5', 'targets')
            size = 19962   
            
        if set_type=='train_cls5':
            X_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_train_5cls.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_train_5cls.hdf5', 'targets')
            size = 122077        
            
        elif set_type=='valid_cls5':
            X_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_valid_5cls.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_valid_5cls.hdf5', 'targets')
            size = 15138        
            
        elif set_type=='test_cls5':
            X_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_test_5cls.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_test_5cls.hdf5', 'targets')
            size = 14724          
            
        elif set_type=='all':
            X_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped.hdf5', 'targets')
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
        i = Conv2D(nop,kernel_size=(kw,kw),padding=bm,strides=(std,std), kernel_initializer='random_uniform')(i)
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
    
    def conv(i,nop,kw,std=1,usebn=True,bm='same'):
        i = Conv2D(nop,kernel_size=(kw,kw),padding=bm,strides=(std,std), kernel_initializer='random_uniform')(i)
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
    i = Flatten()(i)
    i = Dense(200)(i)
    i = Activation('relu',name='relu_dens1')(i)
    # i = Dense(200)(i)
    # i = Activation('relu',name='relu_dens2')(i)
    i = Dense(num_classes)(i)
    predictions=Activation('sigmoid')(i)
    model = Model(inputs=inp, outputs=predictions)  
    
    return model

def gan_dis_model_cel():
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

    # 4 and 5
    # i = conv(i,ndf*1,4,std=2,name='0',usebn=False)
    # i = concat_diff(i)
    # i = conv(i,ndf*2,4,std=2,name='1')
    # i = concat_diff(i)
    # i = conv(i,ndf*4,4,std=2,name='2')
    # i = concat_diff(i)
    # i = conv(i,ndf*8,4,std=2,name='3')
    # i = concat_diff(i)
    # i0 = conv(i,ndf*8,4,std=2,name='4_cel0')
    # i0 = concat_diff(i0)
    # i1 = conv(i,ndf*8,4,std=2,name='4_cel1')
    # i1 = concat_diff(i1)
    # i2 = conv(i,ndf*8,4,std=2,name='4_cel2')
    # i2 = concat_diff(i2)
    # i3 = conv(i,ndf*8,4,std=2,name='4_cel3')
    # i3 = concat_diff(i3)
    # i4 = conv(i,ndf*8,4,std=2,name='4_cel4')
    # i4 = concat_diff(i4)
    # i0 = conv(i0,ndf*8,4,std=2,name='5_cel0')
    # i0 = concat_diff(i0)
    # i0 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel0')(i0)
    # i1 = conv(i1,ndf*8,4,std=2,name='5_cel1')
    # i1 = concat_diff(i1)
    # i1 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel1')(i1)
    # i2 = conv(i2,ndf*8,4,std=2,name='5_cel2')
    # i2 = concat_diff(i2)
    # i2 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel2')(i2)
    # i3 = conv(i3,ndf*8,4,std=2,name='5_cel3')
    # i3 = concat_diff(i3)
    # i3 = Conv2D(39,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel3')(i3)
    # i4 = conv(i4,ndf*8,4,std=2,name='5_cel4')
    # i4 = concat_diff(i4)
    # i4 = Conv2D(39,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel4')(i4)
    # i = Concatenate()([i0,i1,i2,i3,i4])
    # i = concat_diff(i)
    # i = Flatten()(i)
    # i = Dense(200, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='dense1')(i)
    # i = Activation('relu',name='relu_dens1')(i)
    # # i = Dense(200, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='dense2')(i)
    # # i = Activation('relu',name='relu_dens2')(i)
    # i = Dense(num_classes, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='last_dense')(i)
    # predictions=Activation('sigmoid')(i)
    # model = Model(inputs=inp, outputs=predictions)  
    
    # 3, 4, and 5
    # i = conv(i,ndf*1,4,std=2,name='0',usebn=False)
    # i = concat_diff(i)
    # i = conv(i,ndf*2,4,std=2,name='1')
    # i = concat_diff(i)
    # i = conv(i,ndf*4,4,std=2,name='2')
    # i = concat_diff(i)
    # i0 = conv(i,ndf*8,4,std=2,name='3_cel0')
    # i0 = concat_diff(i0)
    # i1 = conv(i,ndf*8,4,std=2,name='3_cel1')
    # i1 = concat_diff(i1)
    # i2 = conv(i,ndf*8,4,std=2,name='3_cel2')
    # i2 = concat_diff(i2)
    # i3 = conv(i,ndf*8,4,std=2,name='3_cel3')
    # i3 = concat_diff(i3)
    # i4 = conv(i,ndf*8,4,std=2,name='3_cel4')
    # i4 = concat_diff(i4)
    # i0 = conv(i0,ndf*8,4,std=2,name='4_cel0')
    # i0 = concat_diff(i0)
    # i1 = conv(i1,ndf*8,4,std=2,name='4_cel1')
    # i1 = concat_diff(i1)
    # i2 = conv(i2,ndf*8,4,std=2,name='4_cel2')
    # i2 = concat_diff(i2)
    # i3 = conv(i3,ndf*8,4,std=2,name='4_cel3')
    # i3 = concat_diff(i3)
    # i4 = conv(i4,ndf*8,4,std=2,name='4_cel4')
    # i4 = concat_diff(i4)
    # i0 = conv(i0,ndf*8,4,std=2,name='5_cel0')
    # i0 = concat_diff(i0)
    # i0 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel0')(i0)
    # i1 = conv(i1,ndf*8,4,std=2,name='5_cel1')
    # i1 = concat_diff(i1)
    # i1 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel1')(i1)
    # i2 = conv(i2,ndf*8,4,std=2,name='5_cel2')
    # i2 = concat_diff(i2)
    # i2 = Conv2D(38,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel2')(i2)
    # i3 = conv(i3,ndf*8,4,std=2,name='5_cel3')
    # i3 = concat_diff(i3)
    # i3 = Conv2D(39,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel3')(i3)
    # i4 = conv(i4,ndf*8,4,std=2,name='5_cel4')
    # i4 = concat_diff(i4)
    # i4 = Conv2D(39,kernel_size=(1,1),padding='valid',strides=(1,1), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv5.5_cel4')(i4)
    # i = Concatenate()([i0,i1,i2,i3,i4])
    # i = concat_diff(i)
    # i = Flatten()(i)
    # i = Dense(200, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='dense1')(i)
    # i = Activation('relu',name='relu_dens1')(i)
    # # i = Dense(200, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='dense2')(i)
    # # i = Activation('relu',name='relu_dens2')(i)
    # i = Dense(num_classes, kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='last_dense')(i)
    # predictions=Activation('sigmoid')(i)
    # model = Model(inputs=inp, outputs=predictions)  

    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

#(X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes, input_shape = load_data('CelebAHDF5')
num_classes, input_shape = load_data_attr('CelebA_cls5')
# build the model

model = gan_dis_model()
# model = gan_dis_model_cel()
filepath="cnn_weights_best.hdf5"
model.load_weights('/home/ubuntuone/Projects/fce_gan/save/cnn/normal_cnn_weights_best.hdf5')
#model = baseline_model_api(input_shape)
model.summary()
# Compile model
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
#opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.97)
#opt = optimizers.SGD(lr=0.045, momentum=0.9, decay=0.9)
model.compile(optimizer=opt,
          loss=losses.binary_crossentropy,
          metrics=['accuracy'])
          
# checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
# csv_logger = CSVLogger('training.log', append=False)
# tensorboard = TensorBoard(log_dir='./tf-logs')
# callbacks_list = [checkpoint, csv_logger, tensorboard]

#IPython.embed()
## Fit the model
#model.fit(X_train[0:162770], y_train[0:162770], validation_data=(X_val[162770:182637], y_val[162770:182637]), epochs=2, batch_size=200, verbose=2)

## Final evaluation of the model
#scores = model.evaluate(X_test[182637:202599], y_test[182637:202599], verbose=0)
#print("Baseline Error: %.2f%%" % (100-scores[1]*100))
#prediction=model.predict(X_test[202599])

# Fit the model with generator
# CelebA data size: train = 162770, valid = 19867, test = 19962
#callbacks = [TensorBoard(log_dir='./tf-logs')]
# result = model.fit_generator(hdf5_generator('CelebA','train_cls5'),611,1000,validation_data=hdf5_generator('CelebA','valid_cls5'),validation_steps=76,callbacks=callbacks_list)

## Final evaluation of the model with generator
# scores = model.evaluate_generator(hdf5_generator('CelebA','test_cls5'),74)
# print("Baseline Error: %.2f%%" % (100-scores[1]*100))
prediction=model.predict_generator(hdf5_generator('CelebA','test_cls5'), 14724)
prediction=numpy.transpose(prediction)
prediction[prediction<0.5]=0
prediction[prediction>=0.5]=1
prediction.astype(int)
prediction_flip=numpy.absolute(1-prediction)

y_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_test_5cls.hdf5', 'targets')
y_data=y_data[:].astype(int)
y_data_flip=numpy.absolute(1-y_data)

confmat=numpy.matmul(prediction, y_data)
confmat_flip=numpy.matmul(prediction_flip, y_data_flip)

ap=numpy.count_nonzero(y_data[:,0])+numpy.count_nonzero(y_data[:,1])+numpy.count_nonzero(y_data[:,2])+numpy.count_nonzero(y_data[:,3])+numpy.count_nonzero(y_data[:,4])
an=numpy.count_nonzero(y_data_flip[:,0])+numpy.count_nonzero(y_data_flip[:,1])+numpy.count_nonzero(y_data_flip[:,2])+numpy.count_nonzero(y_data_flip[:,3])+numpy.count_nonzero(y_data_flip[:,4])
tp=confmat[0,0]+confmat[1,1]+confmat[2,2]+confmat[3,3]+confmat[4,4]
tn=confmat_flip[0,0]+confmat_flip[1,1]+confmat_flip[2,2]+confmat_flip[3,3]+confmat_flip[4,4]
fp=ap-tp
fn=an-tn
precision=tp/ap
recall=tp/(tp+fn)
acc=(tp+tn)/(ap+an)


ap0=numpy.count_nonzero(y_data[:,0])
an0=numpy.count_nonzero(y_data_flip[:,0])
tp0=confmat[0,0]
tn0=confmat_flip[0,0]
fp0=ap0-tp0
fn0=an0-tn0
precision0=tp0/ap0
recall0=tp0/(tp0+fn0)
acc0=(tp0+tn0)/(ap0+an0)

ap1=numpy.count_nonzero(y_data[:,1])
an1=numpy.count_nonzero(y_data_flip[:,1])
tp1=confmat[1,1]
tn1=confmat_flip[1,1]
fp1=ap1-tp1
fn1=an1-tn1
precision1=tp1/ap1
recall1=tp1/(tp1+fn1)
acc1=(tp1+tn1)/(ap1+an1)

ap2=numpy.count_nonzero(y_data[:,2])
an2=numpy.count_nonzero(y_data_flip[:,2])
tp2=confmat[2,2]
tn2=confmat_flip[2,2]
fp2=ap2-tp2
fn2=an2-tn2
precision2=tp2/ap2
recall2=tp2/(tp2+fn2)
acc2=(tp2+tn2)/(ap2+an2)

ap3=numpy.count_nonzero(y_data[:,3])
an3=numpy.count_nonzero(y_data_flip[:,3])
tp3=confmat[3,3]
tn3=confmat_flip[3,3]
fp3=ap3-tp3
fn3=an3-tn3
precision3=tp3/ap3
recall3=tp3/(tp3+fn3)
acc3=(tp3+tn3)/(ap3+an3)

ap4=numpy.count_nonzero(y_data[:,4])
an4=numpy.count_nonzero(y_data_flip[:,4])
tp4=confmat[4,4]
tn4=confmat_flip[4,4]
fp4=ap4-tp4
fn4=an4-tn4
precision4=tp4/ap4
recall4=tp4/(tp4+fn4)
acc4=(tp4+tn4)/(ap4+an4)

IPython.embed()