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
from keras.callbacks import TensorBoard, ModelCheckpoint, CSVLogger, EarlyStopping, LearningRateScheduler, History
from keras.models import load_model


TF_WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels.h5'
TF_WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.4/xception_weights_tf_dim_ordering_tf_kernels_notop.h5'

import numpy
import numpy as np
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
import pickle
import time

#sess = K.get_session()
#sess = tf_debug.LocalCLIDebugWrapperSession(sess)
#K.set_session(sess)
reg_val = 5e-6
homepath = '/mnt/Storage'

def normalize_pixel(data):
    return data/255-.5

def load_data(dataset):
    if dataset=='CelebAHDF5_cls5_val':
        # load data CelebA
        X_data = HDF5Matrix(homepath+'/Projects/data/CelebAHDF5/celeba_aligned_cropped_valid_5cls.hdf5', 'features', normalizer=normalize_pixel)
        y_data = HDF5Matrix(homepath+'/Projects/data/CelebAHDF5/celeba_aligned_cropped_valid_5cls.hdf5', 'targets')
        
        num_classes = y_data.shape[1]
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

def load_bootstrap(train,icls,cnt4):
    cnt=np.where(train['targets'][icls.tolist()][:,2]==1)[0].shape[0]
    cls4_0=train['targets'][:,4]==0
    if cnt4-cnt>0:
        cls2_1=train['targets'][:,2]==1
        icls_sample=np.random.choice(np.where(np.logical_and.reduce((cls4_0, cls2_1)))[0],cnt4-cnt,replace=False)
        icls=np.concatenate((icls,icls_sample))
        icls=np.sort(icls)

    #sample 1
    cnt=np.where(train['targets'][icls.tolist()][:,1]==1)[0].shape[0]
    cls2_0=train['targets'][:,2]==0
    if cnt4-cnt>0:
        cls1_1=train['targets'][:,1]==1
        icls_sample=np.random.choice(np.where(np.logical_and.reduce((cls4_0, cls2_0, cls1_1)))[0],cnt4-cnt,replace=False)
        icls=np.concatenate((icls,icls_sample))
        icls=np.sort(icls)

    #sample 3
    cnt=np.where(train['targets'][icls.tolist()][:,3]==1)[0].shape[0]
    cls1_0=train['targets'][:,1]==0
    if cnt4-cnt>0:
        cls3_1=train['targets'][:,3]==1
        icls_sample=np.random.choice(np.where(np.logical_and.reduce((cls4_0, cls2_0, cls1_0, cls3_1)))[0],cnt4-cnt,replace=False)
        icls=np.concatenate((icls,icls_sample))
        icls=np.sort(icls)

    #sample 0
    cnt=np.where(train['targets'][icls.tolist()][:,0]==1)[0].shape[0]
    cls3_0=train['targets'][:,3]==0
    if cnt4-cnt>0:
        cls0_1=train['targets'][:,0]==1
        icls_sample=np.random.choice(np.where(np.logical_and.reduce((cls4_0, cls2_0, cls1_0, cls3_0, cls0_1)))[0],cnt4-cnt,replace=False)
        icls=np.concatenate((icls,icls_sample))
        # icls=np.sort(icls)

    np.random.shuffle(icls)

    return (icls)

def hdf5_generator(dataset,set_type,indexes=None):
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

        if indexes is not None:
            size = indexes.shape[0]
                    
        while 1:
            if indexes is None:
                X_single = X_data[i%size].reshape((1, 218, 178, 3))
                y_single = y_data[i%size].reshape((1, 5))
            else:
                X_single = X_data[indexes[i%size]].reshape((1, 218, 178, 3))
                y_single = y_data[indexes[i%size]].reshape((1, 5))

            yield(X_single, y_single)
            i+=1

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

(X_val, y_val), num_classes, input_shape = load_data('CelebAHDF5_cls5_val')
# num_classes, input_shape = load_data_attr('CelebA_cls5')

# build the model
model = gan_dis_model()
model.summary()
filepath="cnn_weights_best.hdf5"
filepath_routine="cnn_weights.hdf5"
# model=load_model(filepath)
# Compile model
opt = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-6)
model.compile(optimizer=opt,
          loss=losses.binary_crossentropy,
          metrics=['accuracy'])

checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
checkpoint_routine = ModelCheckpoint(filepath_routine, monitor='val_acc', verbose=1, save_best_only=False, mode='max')
csv_logger = CSVLogger('training.log', append=True)
tensorboard = TensorBoard(log_dir='./tf-logs')
callbacks_list = [checkpoint, checkpoint_routine, csv_logger, tensorboard]

# f=open("history.pickle","rb")
# history = History()
# IPython.embed()
# history.history = pickle.load(f)
# f.close()
# callbacks_list = [checkpoint, csv_logger, tensorboard, history]

f=open("history.pickle","wb")

batch_size=256
train=h5py.File(homepath+'/Projects/data/CelebAHDF5/celeba_aligned_cropped_train_5cls.hdf5','r')
icls4=np.where(train['targets'][:,4]==1)[0]
cnt4=icls4.shape[0]

print('start training')
initial_epoch=298
for i in range(1000-initial_epoch):
    if i+initial_epoch!=0:
        model=load_model(filepath_routine)
    # print('epoch '+str(i+initial_epoch))
    
    # start = time.time()
    icls=load_bootstrap(train,icls4,cnt4)
    # end = time.time()
    # print('bootstrapping for '+str(end-start)+'s')

    # IPython.embed()

    history = model.fit_generator(hdf5_generator('CelebA','train_cls5',icls),icls.shape[0]//batch_size,i+initial_epoch+1,initial_epoch=i+initial_epoch,validation_data=hdf5_generator('CelebA','valid_cls5'),validation_steps=15138//batch_size,callbacks=callbacks_list, max_queue_size=1, workers=1, use_multiprocessing=False)


    callbacks_list = [checkpoint, checkpoint_routine, csv_logger, tensorboard, history]
    # pickle.dump(history.history, f)

# Final evaluation of the model with generator
# scores = model.evaluate_generator(hdf5_generator('CelebA','test_cls5'),74)
# print("Baseline Error: %.2f%%" % (100-scores[1]*100))
# prediction=model.predict_generator(hdf5_generator('CelebA','test_cls5'), 74)

f.close()


# IPython.embed()