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
import tensorflow as tf

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
                                      trainable=self.trainable)
        super(ClassDependentCost, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x):
        return x * self.sigmoid(self.kernel)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_dim)

def concat_diff(i): # batch discrimination -  increase generation diversity.
    # return i
    bv = Lambda(lambda i:K.mean(K.abs(i[:] - K.mean(i,axis=0)),axis=-1,keepdims=True))(i)
    i = Concatenate()([i,bv])
    return i
 
def baseline_model_api(input_shape):
    inputs = Input(shape=input_shape)
    x=Conv2D (32, (5, 5))(inputs)
    x=Activation('relu')(x)
    x=MaxPooling2D(pool_size=(2, 2))(x)
    x=Dropout(0.2)(x)
    x=Flatten()(x)
    x=Dense(128)(x)
    x=Activation('relu')(x)
    x=Dense(num_classes)(x)
    predictions=Activation('softmax')(x)
    model = Model(inputs=inputs, outputs=predictions)    
#    model.compile(optimizer='adam',
#              loss='categorical_crossentropy',
#              metrics=['accuracy'])
    return model

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
    predictions=ClassDependentCost(5,trainable=True)(predictions)
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

model = gan_dis_model()
#model = baseline_model_api(input_shape)
model.summary()
filepath="cnn_weights_best.hdf5"
# model=load_model(filepath)
# Compile model
# opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
#opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.97)
opt = optimizers.SGD(lr=0.045, momentum=0.9, decay=0.9)
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
result = model.fit_generator(hdf5_generator('CelebA','train_cls5'),122077//batch_size,2,initial_epoch=0,validation_data=hdf5_generator('CelebA','valid_cls5'),validation_steps=15138//batch_size,callbacks=callbacks_list)

# Final evaluation of the model with generator
scores = model.evaluate_generator(hdf5_generator('CelebA','test_cls5'),74)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
# prediction=model.predict_generator(hdf5_generator('CelebA','test_cls5'), 74)

output_before_cost = model.layers[34].output
output_after_cost = model.layers[35].output
functor = K.function([model.input]+ [K.learning_phase()], [output_before_cost, output_after_cost])

X = hdf5_generator('CelebA','valid_cls5').__next__()[0]
layer_outs = functor([X, 1.])
weights = model.layers[35].get_weights()

IPython.embed()