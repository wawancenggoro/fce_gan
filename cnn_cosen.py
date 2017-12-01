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
                                      regularizer=regularizers.l2(1e-1),
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
        for i in range(35):
            model.layers[i].trainable = False

    
    return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
#X_data = HDF5Matrix('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped.hdf5', 'features', normalizer=normalize_pixel) 
#IPython.embed()

#(X_train, y_train), (X_val, y_val), (X_test, y_test), num_classes, input_shape = load_data('CelebAHDF5')
num_classes, input_shape = load_data_attr('CelebA_cls5')
# build the model

model = cosen_cnn_model('normal')
model_cost = cosen_cnn_model('cost')
#model = baseline_model_api(input_shape)
model.summary()
model_cost.summary()
filepath="cnn_weights_best.hdf5"
filepath_routine="cnn_weights.hdf5"
filepath_routine_cost="cnn_cost_weights.hdf5"
# model=load_model(filepath)
# Compile model
# opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
#opt = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-08, decay=0.97)
opt = optimizers.SGD(lr=0.001, momentum=0, decay=0)
model.compile(optimizer=opt,
          loss=losses.binary_crossentropy,
          metrics=['accuracy'])
model_cost.compile(optimizer=opt,
          loss=losses.binary_crossentropy,
          metrics=['accuracy'])

def schedule(epoch):
    if epoch<100:
        return epoch*0.001/100
    else:
        return 0.001

          
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint_routine = ModelCheckpoint(filepath_routine, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
csv_logger = CSVLogger('training.log', append=False)
tensorboard = TensorBoard(log_dir='./tf-logs')
lr_schedule = LearningRateScheduler(schedule)
callbacks_list = [checkpoint, checkpoint_routine, csv_logger, tensorboard]

checkpoint_routine_cost = ModelCheckpoint(filepath_routine_cost, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
callbacks_list_cost = [checkpoint_routine_cost]
# Fit the model with generator
# CelebA data size: train = 162770, valid = 19867, test = 19962
# CelebA cls5 data size: train = 122077, valid = 15138, test = 14724
# callbacks = [TensorBoard(log_dir='./tf-logs')]
batch_size=256
initial_epoch = 0
# cost_train_step=122077//batch_size
# cost_valid_step=15138//batch_size
cost_train_step=1
cost_valid_step=1
for i in range(1000):
    if i!=0:
        model_cost.load_weights(filepath_routine)
    history_cost = model_cost.fit_generator(hdf5_generator('CelebA','train_cls5'),cost_train_step,i+initial_epoch+1,initial_epoch=i+initial_epoch,validation_data=hdf5_generator('CelebA','valid_cls5'),validation_steps=cost_valid_step,callbacks=callbacks_list_cost)
    callbacks_list_cost = [checkpoint_routine_cost, history_cost]

    model.load_weights(filepath_routine_cost)
    history = model.fit_generator(hdf5_generator('CelebA','train_cls5'),122077//batch_size,i+initial_epoch+1,initial_epoch=i+initial_epoch,validation_data=hdf5_generator('CelebA','valid_cls5'),validation_steps=15138//batch_size,callbacks=callbacks_list)
    callbacks_list = [checkpoint, checkpoint_routine, csv_logger, tensorboard, history]

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