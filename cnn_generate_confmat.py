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
from keras.engine.topology import Layer
from keras import activations
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

model_homepath = '/home/wawan/git/'
data_homepath = '/mnt/Storage/'
filepath = 'cnn/unfrozen/cel012345_cnn_weights_best.hdf5'
modelname = model_homepath+'fce_gan/save/stat/unfrozen_cel012345'


def normalize_pixel(data):
    return data/255-.5
    
def hdf5_generator(dataset,set_type):
    i=0    
    # CelebA data size: train = 162770, valid = 19867, test = 19962
    if dataset=='CelebA':    
        if set_type=='train':
            X_data = HDF5Matrix(data_homepath+'Projects/data/CelebAHDF5/celeba_aligned_cropped_train.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix(data_homepath+'Projects/data/CelebAHDF5/celeba_aligned_cropped_train.hdf5', 'targets')
            size = 162770        
            
        elif set_type=='valid':
            X_data = HDF5Matrix(data_homepath+'Projects/data/CelebAHDF5/celeba_aligned_cropped_valid.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix(data_homepath+'Projects/data/CelebAHDF5/celeba_aligned_cropped_valid.hdf5', 'targets')
            size = 19867        
            
        elif set_type=='test':
            X_data = HDF5Matrix(data_homepath+'Projects/data/CelebAHDF5/celeba_aligned_cropped_test.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix(data_homepath+'Projects/data/CelebAHDF5/celeba_aligned_cropped_test.hdf5', 'targets')
            size = 19962   
            
        if set_type=='train_cls5':
            X_data = HDF5Matrix(data_homepath+'Projects/data/CelebAHDF5/celeba_aligned_cropped_train_5cls.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix(data_homepath+'Projects/data/CelebAHDF5/celeba_aligned_cropped_train_5cls.hdf5', 'targets')
            size = 122077        
            
        elif set_type=='valid_cls5':
            X_data = HDF5Matrix(data_homepath+'Projects/data/CelebAHDF5/celeba_aligned_cropped_valid_5cls.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix(data_homepath+'Projects/data/CelebAHDF5/celeba_aligned_cropped_valid_5cls.hdf5', 'targets')
            size = 15138        
            
        elif set_type=='test_cls5':
            X_data = HDF5Matrix(data_homepath+'Projects/data/CelebAHDF5/celeba_aligned_cropped_test_5cls.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix(data_homepath+'Projects/data/CelebAHDF5/celeba_aligned_cropped_test_5cls.hdf5', 'targets')
            size = 14724          
            
        elif set_type=='all':
            X_data = HDF5Matrix(data_homepath+'Projects/data/CelebAHDF5/celeba_aligned_cropped.hdf5', 'features', normalizer=normalize_pixel)
            y_data = HDF5Matrix(data_homepath+'Projects/data/CelebAHDF5/celeba_aligned_cropped.hdf5', 'targets')
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

class ClassDependentCost(Layer):

    def __init__(self, output_dim, trainable, **kwargs):
        self.output_dim = output_dim
        super(ClassDependentCost, self).__init__(**kwargs)
        self.sigmoid = activations.get('sigmoid')
        self.trainable = trainable

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.kernel = self.add_weight(name='kernel', 
                                      shape=(input_shape[1],),
                                      initializer='he_uniform',
                                      regularizer=regularizers.l2(reg_val),
                                      trainable=self.trainable)
        super(ClassDependentCost, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.sigmoid(self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def cosen_cnn_model(type):
    inp = Input(shape=(218,178,3))
    i = inp

    ndf=24
    
    def conv(i,nop,kw,std=1,usebn=True,bm='same',name=''):
        i = Conv2D(nop,kernel_size=(kw,kw),padding=bm,strides=(std,std), kernel_initializer='he_uniform', kernel_regularizer=regularizers.l2(reg_val), bias_regularizer=regularizers.l2(reg_val),name='conv'+name)(i)
        if usebn:
            i = BatchNormalization(name='bn'+name)(i)
        i = Activation('relu',name='relu'+name)(i)
        return i
    
    if type=='cost':
        cost_trainable = True
    elif type=='normal':
        cost_trainable = False

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
    i=ClassDependentCost(5,trainable=cost_trainable)(i)
    predictions=Activation('sigmoid')(i)
    model = Model(inputs=inp, outputs=predictions) 

    if type=='cost':
        for i in range(34):
            model.layers[i].trainable = False
    
    return model

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

num_classes, input_shape = load_data_attr('CelebA_cls5')
# build the model

# model = gan_dis_model()
model = gan_dis_model_cel012345()
# model = cosen_cnn_model('normal')
model.load_weights(model_homepath+'fce_gan/save/'+filepath)
model.summary()     

prediction=model.predict_generator(hdf5_generator('CelebA','test_cls5'), 14724)
prediction=numpy.transpose(prediction)
prediction[prediction<0.5]=0
prediction[prediction>=0.5]=1
prediction.astype(int)
prediction_flip=numpy.absolute(1-prediction)

y_data = HDF5Matrix(data_homepath+'Projects/data/CelebAHDF5/celeba_aligned_cropped_test_5cls.hdf5', 'targets')
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

save = numpy.array([
    [ap, ap0, ap1, ap2, ap3, ap4], 
    [an, an0, an1, an2, an3, an4], 
    [tp, tp0, tp1, tp2, tp3, tp4], 
    [tn, tn0, tn1, tn2, tn3, tn4], 
    [fp, fp0, fp1, fp2, fp3, fp4], 
    [fn, fn0, fn1, fn2, fn3, fn4], 
    [precision, precision0, precision1, precision2, precision3, precision4], 
    [recall, recall0, recall1, recall2, recall3, recall4], 
    [acc, acc0, acc1, acc2, acc3, acc4]
])

numpy.savetxt(modelname+'_cnn_stat.csv', save, delimiter=',')

# IPython.embed()