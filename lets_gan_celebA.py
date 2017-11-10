from __future__ import print_function

import tensorflow as tf

from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import *
from keras.layers import *
# from keras.optimizers import *
from keras.utils import np_utils
import keras
import keras.backend as K

import math
import random

import numpy as np

import cv2

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
from keras.layers import Conv2DTranspose
from keras.layers import concatenate
from keras.engine.topology import get_source_inputs
from keras.utils.data_utils import get_file
from keras import backend as K
from keras.applications.imagenet_utils import decode_predictions
from keras.applications.imagenet_utils import _obtain_input_shape

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils
from keras import optimizers
import IPython
import h5py
from keras.utils.io_utils import HDF5Matrix

batch_size = 32
nb_classes = 10
nb_epoch = 200
eps=1e-11

zed = 100

def cifar():
    # input image dimensions
    img_rows, img_cols = 32, 32
    # the CIFAR10 images are RGB
    img_channels = 3

    # the data, shuffled and split between train and test sets
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    # convert class vectors to binary class matrices
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    X_train /= 255
    X_test /= 255

    X_train-=0.5
    X_test-=0.5

    return X_train,Y_train,X_test,Y_test
#print('loading cifar...')
#xt,yt,xv,yv = cifar()

def celebA():
    # input image dimensions
    img_rows, img_cols = 218, 178
    # the CIFAR10 images are RGB
    img_channels = 3

    # the data, shuffled and split between train and test sets
    train=h5py.File('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_train.hdf5','r')
#    valid=h5py.File('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_valid.hdf5','r')
    test=h5py.File('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_test.hdf5','r')

#    # convert class vectors to binary class matrices
#    Y_train = np_utils.to_categorical(y_train, nb_classes)
#    Y_test = np_utils.to_categorical(y_test, nb_classes)
#
#    X_train = X_train.astype('float32')
#    X_test = X_test.astype('float32')
#
#    X_train /= 255
#    X_test /= 255
#
#    X_train-=0.5
#    X_test-=0.5

    return train, test

train_data, test_data = celebA()

def leakyRelu(i):
    return LeakyReLU(.2)(i)

def relu(i):
    return Activation('relu')(i)
    
def bn(i):
    return BatchNormalization()(i)

def gen2(): # generative network, 2
    inp = Input(shape=(zed,))
    i = inp
    i = Reshape((1,1,zed))(i)

    ngf=24

    def deconv(i,nop,kw,oh,ow,std=1,tail=True,bm='same'):
        global batch_size
        i = Deconvolution2D(nop,kw,kw,subsample=(std,std),border_mode=bm,output_shape=(batch_size,oh,ow,nop))(i)
        if tail:
            i = bn(i)
            i = relu(i)
        return i

    i = deconv(i,nop=ngf*8,kw=4,oh=4,ow=4,std=1,bm='valid')
    i = deconv(i,nop=ngf*4,kw=4,oh=8,ow=8,std=2)
    i = deconv(i,nop=ngf*2,kw=4,oh=16,ow=16,std=2)
    i = deconv(i,nop=ngf*1,kw=4,oh=32,ow=32,std=2)

    i = deconv(i,nop=3,kw=4,oh=32,ow=32,std=1,tail=False) # out : 32x32
    i = Activation('tanh')(i)

    m = Model(input=inp,output=i)
    return m

def gen2_keras2(): # generative network, 2
    inp = Input(shape=(zed,))
    i = inp
    i = Reshape((1,1,zed))(i)

    ngf=24

    def deconv(i,nop,kw,oh,ow,std=1,tail=True,bm='same'):
        global batch_size
        i = Conv2DTranspose(nop,kernel_size=(kw,kw),strides=(std,std),padding=bm, kernel_initializer='random_uniform')(i)
        if tail:
            i = bn(i)
            i = relu(i)
        return i

    i = deconv(i,nop=ngf*8,kw=4,oh=4,ow=4,std=1,bm='valid')
    i = deconv(i,nop=ngf*4,kw=4,oh=8,ow=8,std=2)
    i = deconv(i,nop=ngf*2,kw=4,oh=16,ow=16,std=2)
    i = deconv(i,nop=ngf*1,kw=4,oh=32,ow=32,std=2)

    i = deconv(i,nop=3,kw=4,oh=32,ow=32,std=1,tail=False) # out : 32x32
    i = Activation('tanh')(i)

    m = Model(input=inp,output=i)
    return m

def gen2_keras2_celebA(): # generative network, 2
    inp = Input(shape=(zed,))
    i = inp
    i = Reshape((1,1,zed))(i)

    ngf=24

    def deconv(i,nop,kw,kh,std=1,tail=True,bm='same'):
        global batch_size
        i = Conv2DTranspose(nop,kernel_size=(kw,kh),strides=(std,std),padding=bm, kernel_initializer='random_uniform')(i)
        if tail:
            i = bn(i)
            i = relu(i)
        return i

    i = deconv(i,nop=32,kw=5,kh=4,std=1,bm='valid')
    i = deconv(i,nop=32,kw=4,kh=4,std=5)
    i = deconv(i,nop=32,kw=3,kh=3,std=1,bm='valid')
    i = deconv(i,nop=32,kw=4,kh=4,std=2)
    i = deconv(i,nop=32,kw=4,kh=4,std=2)
    i = deconv(i,nop=32,kw=4,kh=4,std=2,bm='valid')

    i = deconv(i,nop=3,kw=4,kh=4,std=1,tail=False) # out : 218x178
    i = Activation('tanh')(i)

    m = Model(input=inp,output=i)
    return m
    
def gen2_mod(): # generative network, 2
    inp = Input(shape=(zed,))
    i = inp
    i = Reshape((1,1,zed))(i)

    ngf=24

    def deconv(i,nop,kw,kh,std=1,tail=True,bm='same'):
        global batch_size
        i = Conv2DTranspose(nop,kernel_size=(kw,kh),strides=(std,std),padding=bm, kernel_initializer='random_uniform')(i)
        if tail:
            i = bn(i)
            i = relu(i)
        return i

    i = deconv(i,nop=ngf*8,kw=5,kh=4,std=1,bm='valid')
    i = deconv(i,nop=ngf*8,kw=4,kh=4,std=5)
    i = deconv(i,nop=ngf*8,kw=3,kh=3,std=1,bm='valid')
    i = deconv(i,nop=ngf*4,kw=4,kh=4,std=2)
    i = deconv(i,nop=ngf*2,kw=4,kh=4,std=2)
    i = deconv(i,nop=ngf*1,kw=4,kh=4,std=2,bm='valid')

#    i = deconv(i,nop=32,kw=5,kh=4,std=1,bm='valid')
#    i = deconv(i,nop=32,kw=4,kh=4,std=5)
#    i = deconv(i,nop=32,kw=3,kh=3,std=1,bm='valid')
#    i = deconv(i,nop=32,kw=4,kh=4,std=2)
#    i = deconv(i,nop=32,kw=4,kh=4,std=2)
#    i = deconv(i,nop=32,kw=4,kh=4,std=2,bm='valid')
    
    i = deconv(i,nop=3,kw=4,kh=4,std=1,tail=False) # out : 218x178
    i = Activation('tanh')(i)

    m = Model(input=inp,output=i)
    return m

def concat_diff(i): # batch discrimination -  increase generation diversity.
    # return i
    bv = Lambda(lambda i:K.mean(K.abs(i[:] - K.mean(i,axis=0)),axis=-1,keepdims=True))(i)
    i = concatenate([i,bv])
    return i

def concat_diff_keras2(i): # batch discrimination -  increase generation diversity.
    # return i
    bv = Lambda(lambda i:K.mean(K.abs(i[:] - K.mean(i,axis=0)),axis=-1,keepdims=True))(i)
    i = concatenate([i,bv])
    return i

def dis2(): # discriminative network, 2
    # inp = Input(shape=(None,None,3))
    inp = Input(shape=(32,32,3))
    i = inp

    ndf=24

    def conv(i,nop,kw,std=1,usebn=True,bm='same'):
        i = Convolution2D(nop,kw,kw,border_mode=bm,subsample=(std,std))(i)
        if usebn:
            i = bn(i)
        i = relu(i)
        return i

    i = conv(i,ndf*1,4,std=2,usebn=False)
    i = concat_diff(i)
    i = conv(i,ndf*2,4,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*4,4,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*8,4,std=2)
    i = concat_diff(i)

    # 1x1
    i = Convolution2D(1,2,2,border_mode='valid')(i)

    i = Activation('linear',name='conv_exit')(i)
    i = Activation('sigmoid')(i)

    i = Reshape((1,))(i)

    m = Model(input=inp,output=i)
    return m
    
def dis2_keras2(): # discriminative network, 2
    # inp = Input(shape=(None,None,3))
#    inp = Input(shape=(32,32,3))
    inp = Input(shape=(218,178,3))
    i = inp

    ndf=24

    def conv(i,nop,kw,std=1,usebn=True,bm='same'):
        i = Conv2D(nop,kernel_size=(kw,kw),padding=bm,strides=(std,std), use_bias=False, kernel_initializer='random_uniform')(i)
        if usebn:
            i = bn(i)
        i = leakyRelu(i)
        return i

    i = conv(i,ndf*1,3,std=2,usebn=False)
    i = concat_diff(i)
    i = conv(i,ndf*2,3,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*4,3,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*8,4,std=2)
    i = concat_diff(i)

    # 1x1
    i = Conv2D(1,(2,2),padding='valid')(i)

    i = Activation('linear',name='conv_exit')(i)
    i = Activation('sigmoid')(i)

    i = Reshape((1,))(i)

    m = Model(input=inp,output=i)
    return m

def dis2_keras2_celebA(): # discriminative network, 2
    # inp = Input(shape=(None,None,3))
#    inp = Input(shape=(32,32,3))
    inp = Input(shape=(218,178,3))
    i = inp

    ndf=24

    def conv(i,nop,kw,std=1,usebn=True,bm='same'):
        i = Conv2D(nop,kernel_size=(kw,kw),padding=bm,strides=(std,std), use_bias=False, kernel_initializer='random_uniform')(i)
        if usebn:
            i = bn(i)
        i = leakyRelu(i)
        return i

    i = conv(i,ndf*1,3,std=2,usebn=False)
    i = concat_diff(i)
    i = conv(i,ndf*2,3,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*4,3,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*8,3,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*8,3,std=2)
    i = concat_diff(i)
    i = conv(i,ndf*8,3,std=2)
    i = concat_diff(i)
    

    # 1x1
    i = Conv2D(1,(2,2),padding='valid')(i)

    i = Activation('linear',name='conv_exit')(i)
    i = Activation('sigmoid')(i)

#    i = Reshape((1,))(i)

    m = Model(input=inp,output=i)
    return m
    
def dis2_mod(): # discriminative network, 2
    # inp = Input(shape=(None,None,3))
#    inp = Input(shape=(32,32,3))
    inp = Input(shape=(218,178,3))
    i = inp

    ndf=24
    
    def conv(i,nop,kw,std=1,usebn=True,bm='same'):
        i = Conv2D(nop,kernel_size=(kw,kw),padding=bm,strides=(std,std), kernel_initializer='random_uniform')(i)
        if usebn:
            i = bn(i)
        i = relu(i)
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
    
#    i = Conv2D(32, (3, 3), strides=(2, 2), padding='same', use_bias=False, name='block1_conv1', kernel_initializer='random_uniform')(i)
#    i = BatchNormalization(name='block1_conv1_bn')(i)
#    i = Activation('relu', name='block1_conv1_act')(i)
#    i = concat_diff(i)
#    i = Conv2D(32, (3, 3), padding='same', use_bias=False, name='block1_conv2', kernel_initializer='random_uniform')(i)
#    i = BatchNormalization(name='block1_conv2_bn')(i)
#    i = Activation('relu', name='block1_conv2_act')(i)
#    i = concat_diff(i)
    
#    i = Conv2D(128, (3, 3), padding='same', use_bias=False, name='block2_conv1', kernel_initializer='random_uniform')(i)
#    i = BatchNormalization(name='block2_conv1_bn')(i)
#    i = Activation('relu', name='block2_conv1_act')(i)
#    i = concat_diff(i)
#    i = Conv2D(128, (3, 3), padding='same', use_bias=False, name='block2_conv2', kernel_initializer='random_uniform')(i)
#    i = BatchNormalization(name='block2_conv2_bn')(i)
#    i = Activation('relu', name='block2_conv2_act')(i)
#    i = concat_diff(i)
    
#    i = Conv2D(256, (3, 3), padding='same', use_bias=False, name='block3_conv1', kernel_initializer='random_uniform')(i)
#    i = BatchNormalization(name='block3_conv1_bn')(i)
#    i = Activation('relu', name='block3_conv1_act')(i)
#    i = concat_diff(i)
#    i = Conv2D(256, (3, 3), padding='same', use_bias=False, name='block3_conv2', kernel_initializer='random_uniform')(i)
#    i = BatchNormalization(name='block3_conv2_bn')(i)
#    i = Activation('relu', name='block3_conv2_act')(i)
#    i = concat_diff(i)
#    
#    i = Conv2D(728, (3, 3), padding='same', use_bias=False, name='block4_conv1', kernel_initializer='random_uniform')(i)
#    i = BatchNormalization(name='block4_conv1_bn')(i)
#    i = Activation('relu', name='block4_conv1_act')(i)
#    i = concat_diff(i)
#    i = Conv2D(728, (3, 3), padding='same', use_bias=False, name='block4_conv2', kernel_initializer='random_uniform')(i)
#    i = BatchNormalization(name='block4_conv2_bn')(i)
#    i = Activation('relu', name='block4_conv2_act')(i)
#    i = concat_diff(i)
    
#    residual = Conv2D(32, (2, 2), strides=(1, 1),
#                      padding='same', use_bias=False, kernel_initializer='random_uniform')(i)
#    residual = BatchNormalization()(residual)
#
#    i = SeparableConv2D(32, (3, 3), padding='same', use_bias=False, name='block2_sepconv1', kernel_initializer='random_uniform')(i)
#    i = BatchNormalization(name='block2_sepconv1_bn')(i)
#    i = Activation('relu', name='block2_sepconv2_act')(i)
#    i = concat_diff(i)
#    i = SeparableConv2D(32, (3, 3), padding='same', use_bias=False, name='block2_sepconv2', kernel_initializer='random_uniform')(i)
#    i = BatchNormalization(name='block2_sepconv2_bn')(i)

#    i = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block2_pool')(i)
#    i = layers.add([i, residual])
#    
#    residual = Conv2D(32, (1, 1), strides=(1, 1),
#                      padding='same', use_bias=False, kernel_initializer='random_uniform')(i)
#    residual = BatchNormalization()(residual)
#
#    i = Activation('relu', name='block3_sepconv1_act')(i)
#    i = concat_diff(i)
#    i = SeparableConv2D(32, (3, 3), padding='same', use_bias=False, name='block3_sepconv1', kernel_initializer='random_uniform')(i)
#    i = BatchNormalization(name='block3_sepconv1_bn')(i)
#    i = Activation('relu', name='block3_sepconv2_act')(i)
#    i = concat_diff(i)
#    i = SeparableConv2D(32, (3, 3), padding='same', use_bias=False, name='block3_sepconv2', kernel_initializer='random_uniform')(i)
#    i = BatchNormalization(name='block3_sepconv2_bn')(i)

#    i = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block3_pool')(i)
#    i = layers.add([i, residual])
#
#    residual = Conv2D(728, (1, 1), strides=(2, 2),
#                      padding='same', use_bias=False, kernel_initializer='random_uniform')(i)
#    residual = BatchNormalization()(residual)
#
#    i = Activation('relu', name='block4_sepconv1_act')(i)
#    i = concat_diff(i)
#    i = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv1', kernel_initializer='random_uniform')(i)
#    i = BatchNormalization(name='block4_sepconv1_bn')(i)
#    i = Activation('relu', name='block4_sepconv2_act')(i)
#    i = concat_diff(i)
#    i = SeparableConv2D(728, (3, 3), padding='same', use_bias=False, name='block4_sepconv2', kernel_initializer='random_uniform')(i)
#    i = BatchNormalization(name='block4_sepconv2_bn')(i)
#
#    i = MaxPooling2D((3, 3), strides=(2, 2), padding='same', name='block4_pool')(i)
#    i = layers.add([i, residual])
#    
#    i = concat_diff(i)
    
#    # 1x1
    i = Conv2D(1,(2,2),padding='valid')(i)

    i = Activation('linear',name='conv_exit')(i)
    i = Activation('sigmoid')(i)

#    i = Reshape((1,))(i)

    m = Model(input=inp,output=i)
    return m

print('generating G...')
#gm = gen2_keras2()
#gm = gen2_keras2_celebA()
gm = gen2_mod()
gm.summary()

print('generating D...')
#dm = dis2_keras2()
#dm = dis2_keras2_celebA()
dm = dis2_mod()
dm.summary()

def gan(g,d):
    # initialize a GAN trainer

    # this is the fastest way to train a GAN in Keras
    # two models are updated simutaneously in one pass

    noise = Input(shape=g.input_shape[1:])
    real_data = Input(shape=d.input_shape[1:])

    generated = g(noise)
    gscore = d(generated)
    rscore = d(real_data)

    def log_eps(i):
        return K.log(i+1e-11)

    # single side label smoothing: replace 1.0 with 0.9
    dloss = - K.mean(log_eps(1-gscore) + .1 * log_eps(1-rscore) + .9 * log_eps(rscore))
    gloss = - K.mean(log_eps(gscore))

    Adam = tf.train.AdamOptimizer

    lr,b1 = 1e-4,.2 # otherwise won't converge.
    optimizer = Adam(lr,beta1=b1)

    grad_loss_wd = optimizer.compute_gradients(dloss, d.trainable_weights)
    update_wd = optimizer.apply_gradients(grad_loss_wd)

    grad_loss_wg = optimizer.compute_gradients(gloss, g.trainable_weights)
    update_wg = optimizer.apply_gradients(grad_loss_wg)

    def get_internal_updates(model):
        # get all internal update ops (like moving averages) of a model
        inbound_nodes = model.inbound_nodes
        input_tensors = []
        for ibn in inbound_nodes:
            input_tensors+= ibn.input_tensors
        updates = [model.get_updates_for(i) for i in input_tensors]
        return updates

    other_parameter_updates = [get_internal_updates(m) for m in [d,g]]
    # those updates includes batch norm.

    print('other_parameter_updates for the models(mainly for batch norm):')
    print(other_parameter_updates)

    train_step = [update_wd, update_wg, other_parameter_updates]
    losses = [dloss,gloss]

    learning_phase = K.learning_phase()

    def gan_feed(sess,batch_image,z_input):
        # actual GAN trainer
        nonlocal train_step,losses,noise,real_data,learning_phase

        res = sess.run([train_step,losses],feed_dict={
        noise:z_input,
        real_data:batch_image,
        learning_phase:True,
        # Keras layers needs to know whether
        # this run is training or testring (you know, batch norm and dropout)
        })

        loss_values = res[1]
        return loss_values #[dloss,gloss]

    return gan_feed

print('generating GAN...')
gan_feed = gan(gm,dm)

print('Ready. enter r() to train')

def r(ep=10000,noise_level=.01):
    sess = K.get_session()

    np.random.shuffle(xt)
    shuffled_cifar = xt
    length = len(shuffled_cifar)

    for i in range(ep):
        noise_level *= 0.99
        print('---------------------------')
        print('iter',i,'noise',noise_level)

        # sample from cifar
        j = i % int(length/batch_size)
        minibatch = shuffled_cifar[j*batch_size:(j+1)*batch_size]
        # minibatch += np.random.normal(loc=0.,scale=noise_level,size=subset_cifar.shape)

        z_input = np.random.normal(loc=0.,scale=1.,size=(batch_size,zed))

        # train for one step
        losses = gan_feed(sess,minibatch,z_input)
        print('dloss:{:6.4f} gloss:{:6.4f}'.format(losses[0],losses[1]))

        if i==ep-1 or i % 10==0: show()

def r_celebA(class_nbr, ep=10000,noise_level=.01,last_ep=0,dm_weights=None,gm_weights=None,save=None):
    sess = K.get_session()

#    np.random.shuffle(xt)
#    shuffled_cifar = xt
#    length = 162770
#    length = 3713
#    length = 50000
#    length = 10000
    
#    class_nbr=4
    if dm_weights is not None:
        dm.load_weights(dm_weights)
    if gm_weights is not None:
        gm.load_weights(gm_weights)
    
    idx = train_data['targets'][:,class_nbr].nonzero()[0].tolist()

    for i in range(ep-last_ep):
        show(save+'.png')
        for j in range(batch_size):
#            noise_level *= 0.99
            print('---------------------------')
            print('iter',i+last_ep+1,'batch',j+1)
    
            # sample from cifar
#            j = i % int(len(idx)/batch_size)
    #        IPython.embed()
            minibatch = (train_data['features'][idx[j*batch_size:(j+1)*batch_size]]/255)-0.5
            # minibatch += np.random.normal(loc=0.,scale=noise_level,size=subset_cifar.shape)
    
            z_input = np.random.normal(loc=0.,scale=1.,size=(batch_size,zed))
    
            # train for one step
            losses = gan_feed(sess,minibatch,z_input)
            print('dloss:{:6.4f} gloss:{:6.4f}'.format(losses[0],losses[1]))

#        if i==ep-1 or i % 10==0: 
#            show()
        if (i+last_ep+1) % 10==0:
#            show(save+'.png')
            if save is not None:
                dm.save_weights('dm_'+save+'.hdf5')
                gm.save_weights('gm_'+save+'.hdf5')

def r_celebA_shuffle(class_nbr, ep=10000,noise_level=.01,last_ep=0,dm_weights=None,gm_weights=None,save=None):
    sess = K.get_session()

#    np.random.shuffle(xt)
#    shuffled_cifar = xt
#    length = 162770
#    length = 3713
#    length = 50000
#    length = 10000
    
#    class_nbr=4
    if dm_weights is not None:
        dm.load_weights(dm_weights)
    if gm_weights is not None:
        gm.load_weights(gm_weights)
    
    idx = train_data['targets'][:,class_nbr].nonzero()[0]

    for i in range(ep-last_ep):
        show(save+'.png')
        for j in range(batch_size):
#            noise_level *= 0.99
            print('---------------------------')
            print('iter',i+last_ep+1,'batch',j+1)
    
            # sample from cifar
            np.random.shuffle(idx)
#            j = i % int(len(idx)/batch_size)
#            IPython.embed()
            idx_batch=idx[j*batch_size:(j+1)*batch_size]
            idx_batch=np.sort(idx_batch).tolist()
#            IPython.embed()
            minibatch = (train_data['features'][idx_batch]/255)-0.5
            # minibatch += np.random.normal(loc=0.,scale=noise_level,size=subset_cifar.shape)
    
            z_input = np.random.normal(loc=0.,scale=1.,size=(batch_size,zed))
    
            # train for one step
            losses = gan_feed(sess,minibatch,z_input)
            print('dloss:{:6.4f} gloss:{:6.4f}'.format(losses[0],losses[1]))

#        if i==ep-1 or i % 10==0: 
#            show()
        if (i+last_ep+1) % 10==0:
#            show(save+'.png')
            if save is not None:
                dm.save_weights('dm_'+save+'.hdf5')
                gm.save_weights('gm_'+save+'.hdf5')



def autoscaler(img):
    limit = 400.
    # scales = [0.1,0.125,1./6.,0.2,0.25,1./3.,1./2.] + range(100)
    scales = np.hstack([1./np.linspace(10,2,num=9), np.linspace(1,100,num=100)])

    imgscale = limit/float(img.shape[0])
#    imgscale = 1
    for s in scales:
        if s>=imgscale:
            imgscale=s
            break
#    IPython.embed()
    img = cv2.resize(img,dsize=(int(img.shape[0]*imgscale),int(img.shape[1]*imgscale)),interpolation=cv2.INTER_NEAREST)

    return img,imgscale

def flatten_multiple_image_into_image(arr):
    import cv2
    num,uh,uw,depth = arr.shape

    patches = int(num+1)
#    height = int(math.sqrt(patches)*0.9)
#    width = int(patches/height+1)
    
    height = 3
    width = 5
    
    img = np.zeros((height*uh+height, width*uw+width, 3),dtype='float32')

    index = 0
    for row in range(height):
        for col in range(width):
            if index>=num-1:
                break
            channels = arr[index]
            img[row*uh+row:row*uh+uh+row,col*uw+col:col*uw+uw+col,:] = channels
            index+=1

#    img,imgscale = autoscaler(img)
    imgscale=1

    return img,imgscale

def show(save=False):
    i = np.random.normal(loc=0.,scale=1.,size=(16,zed))
    gened = gm.predict([i])
#    IPython.embed()

    gened *= 0.5
#    IPython.embed()
    gened +=0.5
#    IPython.embed()

    im,ims = flatten_multiple_image_into_image(gened)
#    IPython.embed()
    
    if save==False:
        cv2.imshow('gened scale:'+str(ims),im[...,::-1])
        cv2.waitKey(1)
    if save!=False:
        cv2.imwrite(save,im[...,::-1]*255)

def save_image(save=False):
    i = np.random.normal(loc=0.,scale=1.,size=(16,zed))
    gened = gm.predict([i])
#    IPython.embed()

    gened *= 0.5
#    IPython.embed()
    gened +=0.5
#    IPython.embed()

    im,ims = flatten_multiple_image_into_image(gened)
#    IPython.embed()
    cv2.imshow('gened scale:'+str(ims),im[...,::-1])
    cv2.waitKey(1)

    if save!=False:
        cv2.imwrite(save,im[...,::-1]*255)

#show('test2.png')
#r_celebA(class_nbr=0, ep=10000, last_ep=10000, dm_weights='dm_fce_0.hdf5', gm_weights='gm_fce_0.hdf5', save='fce_0')
#r_celebA(class_nbr=1, ep=10000, last_ep=10000, dm_weights='dm_fce_1.hdf5', gm_weights='gm_fce_1.hdf5', save='fce_1')
#r_celebA(class_nbr=2, ep=10000, last_ep=10000, dm_weights='dm_fce_2.hdf5', gm_weights='gm_fce_2.hdf5', save='fce_2')
r_celebA(class_nbr=3, ep=10000, last_ep=9810, dm_weights='dm_fce_3.hdf5', gm_weights='gm_fce_3.hdf5', save='fce_3')
#r_celebA(class_nbr=4, ep=10000, last_ep=10000, dm_weights='dm_fce_4.hdf5', gm_weights='gm_fce_4.hdf5', save='fce_4')
#r_celebA(class_nbr=4, ep=10000, last_ep=0, dm_weights=None, gm_weights=None, save='fce_4')
#r_celebA(class_nbr=5, ep=10000, last_ep=1120, dm_weights='dm_fce_5.hdf5', gm_weights='gm_fce_5.hdf5', save='fce_5')
#dm.save_weights('')
#show('test2.png')
