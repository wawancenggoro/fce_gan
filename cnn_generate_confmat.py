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
    
def baseline_model():
	# create model
	model = Sequential()
	model.add(Conv2D(32, (5, 5), input_shape=(28, 28, 1), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
#	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
 
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

def Xception(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=10):
    """Instantiates the Xception architecture.
    Optionally loads weights pre-trained
    on ImageNet. This model is available for TensorFlow only,
    and can only be used with inputs following the TensorFlow
    data format `(width, height, channels)`.
    You should set `image_data_format="channels_last"` in your Keras config
    located at ~/.keras/keras.json.
    Note that the default input image size for this model is 299x299.
    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(299, 299, 3)`.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 71.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.
    # Returns
        A Keras model instance.
    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
        RuntimeError: If attempting to run this model with a
            backend that does not support separable convolutions.
    """
    if weights not in {'imagenet', None}:
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization) or `imagenet` '
                         '(pre-training on ImageNet).')

    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')

    if K.backend() != 'tensorflow':
        raise RuntimeError('The Xception model is only available with '
                           'the TensorFlow backend.')
    if K.image_data_format() != 'channels_last':
        warnings.warn('The Xception model is only available for the '
                      'input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height). '
                      'You should set `image_data_format="channels_last"` in your Keras '
                      'config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    # Determine proper input shape
#    input_shape = _obtain_input_shape(input_shape,
#                                      default_size=299,
#                                      min_size=71,
#                                      data_format=K.image_data_format(),
#                                      require_flatten=False,
#                                      weights=weights)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    x = Conv2D(32, (3, 3), strides=(2, 2), use_bias=False, name='block1_conv1')(img_input)
    x = BatchNormalization(name='block1_conv1_bn')(x)
    x = Activation('relu', name='block1_conv1_act')(x)
    x = Conv2D(64, (3, 3), use_bias=False, name='block1_conv2')(x)
    x = BatchNormalization(name='block1_conv2_bn')(x)
    x = Activation('relu', name='block1_conv2_act')(x)

    residual = Conv2D(128, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv1')(x)
    x = BatchNormalization(name='block2_sepconv1_bn')(x)
    x = Activation('relu', name='block2_sepconv2_act')(x)
    x = SeparableConv2D(128, (3, 3), padding='same', use_bias=False, name='block2_sepconv2')(x)
    x = BatchNormalization(name='block2_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(256, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block3_sepconv1_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv1')(x)
    x = BatchNormalization(name='block3_sepconv1_bn')(x)
    x = Activation('relu', name='block3_sepconv2_act')(x)
    x = SeparableConv2D(256, (3, 3), padding='same', use_bias=False, name='block3_sepconv2')(x)
    x = BatchNormalization(name='block3_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(x)
    x = layers.add([x, residual])

    residual = Conv2D(728, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block4_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1')(x)
    x = BatchNormalization(name='block4_sepconv1_bn')(x)
    x = Activation('relu', name='block4_sepconv2_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2')(x)
    x = BatchNormalization(name='block4_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(x)
    x = layers.add([x, residual])

    for i in range(8):
        residual = x
        prefix = 'block' + str(i + 5)

        x = Activation('relu', name=prefix + '_sepconv1_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv1')(x)
        x = BatchNormalization(name=prefix + '_sepconv1_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv2_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv2')(x)
        x = BatchNormalization(name=prefix + '_sepconv2_bn')(x)
        x = Activation('relu', name=prefix + '_sepconv3_act')(x)
        x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name=prefix + '_sepconv3')(x)
        x = BatchNormalization(name=prefix + '_sepconv3_bn')(x)

        x = layers.add([x, residual])

    residual = Conv2D(1024, (1, 1), strides=(2, 2),
                      padding='same', use_bias=False)(x)
    residual = BatchNormalization()(residual)

    x = Activation('relu', name='block13_sepconv1_act')(x)
    x = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block13_sepconv1')(x)
    x = BatchNormalization(name='block13_sepconv1_bn')(x)
    x = Activation('relu', name='block13_sepconv2_act')(x)
    x = SeparableConv2D(1024, (3, 3), padding='same', use_bias=False, name='block13_sepconv2')(x)
    x = BatchNormalization(name='block13_sepconv2_bn')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block13_pool')(x)
    x = layers.add([x, residual])

    x = SeparableConv2D(1536, (3, 3), padding='same', use_bias=False, name='block14_sepconv1')(x)
    x = BatchNormalization(name='block14_sepconv1_bn')(x)
    x = Activation('relu', name='block14_sepconv1_act')(x)

    x = SeparableConv2D(2048, (3, 3), padding='same', use_bias=False, name='block14_sepconv2')(x)
    x = BatchNormalization(name='block14_sepconv2_bn')(x)
    x = Activation('relu', name='block14_sepconv2_act')(x)

    if include_top:
        x = GlobalAveragePooling2D(name='avg_pool')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input
    # Create model.
    model = Model(inputs, x, name='xception')

    # load weights
    if weights == 'imagenet':
        if include_top:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels.h5',
                                    TF_WEIGHTS_PATH,
                                    cache_subdir='models')
        else:
            weights_path = get_file('xception_weights_tf_dim_ordering_tf_kernels_notop.h5',
                                    TF_WEIGHTS_PATH_NO_TOP,
                                    cache_subdir='models')
        model.load_weights(weights_path)

    if old_data_format:
        K.set_image_data_format(old_data_format)
    return model

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
        i = Conv2D(nop,kernel_size=(kw,kw),padding=bm,strides=(std,std), kernel_initializer='random_uniform',name='conv'+name)(i)
        if usebn:
            i = BatchNormalization(name='bn'+name)(i)
        i = Activation('relu',name='relu'+name)(i)
        return i
    
    # only 5
    # i = conv(i,ndf*1,4,std=2,name='0',usebn=False)
    # i = concat_diff(i)
    # i = conv(i,ndf*2,4,std=2,name='1')
    # i = concat_diff(i)
    # i = conv(i,ndf*4,4,std=2,name='2')
    # i = concat_diff(i)
    # i = conv(i,ndf*8,4,std=2,name='3')
    # i = concat_diff(i)
    # i = conv(i,ndf*8,4,std=2,name='4')
    # i = concat_diff(i)
    # i0 = conv(i,ndf*8,4,std=2,name='5_cel0')
    # i1 = conv(i,ndf*8,4,std=2,name='5_cel1')
    # i2 = conv(i,ndf*8,4,std=2,name='5_cel2')
    # i3 = conv(i,ndf*8,4,std=2,name='5_cel3')
    # i4 = conv(i,ndf*8,4,std=2,name='5_cel4')
    # i = Average()([i0,i1,i2,i3,i4])
    # i = concat_diff(i)
    # i = Flatten()(i)
    # i = Dense(200)(i)
    # i = Activation('relu',name='relu_dens1')(i)
    # i = Dense(200)(i)
    # i = Activation('relu',name='relu_dens2')(i)
    # i = Dense(num_classes)(i)
    # predictions=Activation('sigmoid')(i)
    # model = Model(inputs=inp, outputs=predictions)  

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
    i1 = conv(i,ndf*8,4,std=2,name='4_cel1')
    i2 = conv(i,ndf*8,4,std=2,name='4_cel2')
    i3 = conv(i,ndf*8,4,std=2,name='4_cel3')
    i4 = conv(i,ndf*8,4,std=2,name='4_cel4')
    i = Average()([i0,i1,i2,i3,i4])
    i = concat_diff(i)
    i0 = conv(i,ndf*8,4,std=2,name='5_cel0')
    i1 = conv(i,ndf*8,4,std=2,name='5_cel1')
    i2 = conv(i,ndf*8,4,std=2,name='5_cel2')
    i3 = conv(i,ndf*8,4,std=2,name='5_cel3')
    i4 = conv(i,ndf*8,4,std=2,name='5_cel4')
    i = Average()([i0,i1,i2,i3,i4])
    i = concat_diff(i)
    i=Flatten()(i)
    i = Dense(200)(i)
    i = Activation('relu',name='relu_dens1')(i)
    # i = Dense(200)(i)
    # i = Activation('relu',name='relu_dens2')(i)
    i = Dense(num_classes)(i)
    predictions=Activation('sigmoid')(i)
    model = Model(inputs=inp, outputs=predictions)  
    
    # 3, 4, and 5
    # i = conv(i,ndf*1,4,std=2,name='0',usebn=False)
    # i = concat_diff(i)
    # i = conv(i,ndf*2,4,std=2,name='1')
    # i = concat_diff(i)
    # i = conv(i,ndf*4,4,std=2,name='2')
    # i = concat_diff(i)
    # i0 = conv(i,ndf*8,4,std=2,name='3_cel0')
    # i1 = conv(i,ndf*8,4,std=2,name='3_cel1')
    # i2 = conv(i,ndf*8,4,std=2,name='3_cel2')
    # i3 = conv(i,ndf*8,4,std=2,name='3_cel3')
    # i4 = conv(i,ndf*8,4,std=2,name='3_cel4')
    # i = Average()([i0,i1,i2,i3,i4])
    # i = concat_diff(i)
    # i0 = conv(i,ndf*8,4,std=2,name='4_cel0')
    # i1 = conv(i,ndf*8,4,std=2,name='4_cel1')
    # i2 = conv(i,ndf*8,4,std=2,name='4_cel2')
    # i3 = conv(i,ndf*8,4,std=2,name='4_cel3')
    # i4 = conv(i,ndf*8,4,std=2,name='4_cel4')
    # i = Average()([i0,i1,i2,i3,i4])
    # i = concat_diff(i)
    # i0 = conv(i,ndf*8,4,std=2,name='5_cel0')
    # i1 = conv(i,ndf*8,4,std=2,name='5_cel1')
    # i2 = conv(i,ndf*8,4,std=2,name='5_cel2')
    # i3 = conv(i,ndf*8,4,std=2,name='5_cel3')
    # i4 = conv(i,ndf*8,4,std=2,name='5_cel4')
    # i = Average()([i0,i1,i2,i3,i4])
    # i = concat_diff(i)
    # i=Flatten()(i)
    # i=Dense(num_classes)(i)
    # predictions=Activation('sigmoid')(i)
    # model = Model(inputs=inp, outputs=predictions)  

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
# model0 = gan_dis_model_original()
# model0.load_weights('/home/wawan/git/fce_gan/save/dm_fce_0.hdf5')

# model1 = gan_dis_model_original()
# model1.load_weights('/home/wawan/git/fce_gan/save/dm_fce_1.hdf5')

# model2 = gan_dis_model_original()
# model2.load_weights('/home/wawan/git/fce_gan/save/dm_fce_2.hdf5')

# model3 = gan_dis_model_original()
# model3.load_weights('/home/wawan/git/fce_gan/save/dm_fce_3.hdf5')

# model4 = gan_dis_model_original()
# model4.load_weights('/home/wawan/git/fce_gan/save/dm_fce_4.hdf5')

# model = gan_dis_model_cel()
# IPython.embed()

# only 5
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

# model.layers[20].trainable = False
# model.layers[21].trainable = False
# model.layers[22].trainable = False
# model.layers[23].trainable = False
# model.layers[24].trainable = False
# model.layers[25].trainable = False
# model.layers[26].trainable = False
# model.layers[27].trainable = False
# model.layers[28].trainable = False
# model.layers[29].trainable = False

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

# model.layers[38].set_weights(weights_conv5_cel0)
# model.layers[39].set_weights(weights_conv5_cel1)
# model.layers[40].set_weights(weights_conv5_cel2)
# model.layers[41].set_weights(weights_conv5_cel3)
# model.layers[42].set_weights(weights_conv5_cel4)
# model.layers[43].set_weights(weights_bn5_cel0)
# model.layers[44].set_weights(weights_bn5_cel1)
# model.layers[45].set_weights(weights_bn5_cel2)
# model.layers[46].set_weights(weights_bn5_cel3)
# model.layers[47].set_weights(weights_bn5_cel4)

# model.layers[38].trainable = False
# model.layers[39].trainable = False
# model.layers[40].trainable = False
# model.layers[41].trainable = False
# model.layers[42].trainable = False
# model.layers[43].trainable = False
# model.layers[44].trainable = False
# model.layers[45].trainable = False
# model.layers[46].trainable = False
# model.layers[47].trainable = False


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

# model.layers[15].trainable = False
# model.layers[16].trainable = False
# model.layers[17].trainable = False
# model.layers[18].trainable = False
# model.layers[19].trainable = False
# model.layers[20].trainable = False
# model.layers[21].trainable = False
# model.layers[22].trainable = False
# model.layers[23].trainable = False
# model.layers[24].trainable = False

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

# model.layers[33].set_weights(weights_conv4_cel0)
# model.layers[34].set_weights(weights_conv4_cel1)
# model.layers[35].set_weights(weights_conv4_cel2)
# model.layers[36].set_weights(weights_conv4_cel3)
# model.layers[37].set_weights(weights_conv4_cel4)
# model.layers[38].set_weights(weights_bn4_cel0)
# model.layers[39].set_weights(weights_bn4_cel1)
# model.layers[40].set_weights(weights_bn4_cel2)
# model.layers[41].set_weights(weights_bn4_cel3)
# model.layers[42].set_weights(weights_bn4_cel4)

# model.layers[33].trainable = False
# model.layers[34].trainable = False
# model.layers[35].trainable = False
# model.layers[36].trainable = False
# model.layers[37].trainable = False
# model.layers[38].trainable = False
# model.layers[39].trainable = False
# model.layers[40].trainable = False
# model.layers[41].trainable = False
# model.layers[42].trainable = False

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

# model.layers[51].set_weights(weights_conv5_cel0)
# model.layers[52].set_weights(weights_conv5_cel1)
# model.layers[53].set_weights(weights_conv5_cel2)
# model.layers[54].set_weights(weights_conv5_cel3)
# model.layers[55].set_weights(weights_conv5_cel4)
# model.layers[56].set_weights(weights_bn5_cel0)
# model.layers[57].set_weights(weights_bn5_cel1)
# model.layers[58].set_weights(weights_bn5_cel2)
# model.layers[59].set_weights(weights_bn5_cel3)
# model.layers[60].set_weights(weights_bn5_cel4)

# model.layers[51].trainable = False
# model.layers[52].trainable = False
# model.layers[53].trainable = False
# model.layers[54].trainable = False
# model.layers[55].trainable = False
# model.layers[56].trainable = False
# model.layers[57].trainable = False
# model.layers[58].trainable = False
# model.layers[59].trainable = False
# model.layers[60].trainable = False

model = gan_dis_model()
filepath="cnn_weights_best.hdf5"
model.load_weights('/home/wawan/git/fce_gan/save/glorot_normal/normal_dens2_l2glorot_cnn_weights_best.hdf5')
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
scores = model.evaluate_generator(hdf5_generator('CelebA','test_cls5'),74)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
prediction=model.predict_generator(hdf5_generator('CelebA','test_cls5'), 74)


IPython.embed()