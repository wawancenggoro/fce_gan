# -*- coding: utf-8 -*-
"""
Created on Thu Sep 21 14:03:53 2017

@author: wawan
"""

import IPython
import h5py
import numpy as np
from keras.utils.io_utils import HDF5Matrix

train=h5py.File('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_train.hdf5','r')
valid=h5py.File('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_valid.hdf5','r')
test=h5py.File('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped_test.hdf5','r')
#alldata=h5py.File('/mnt/Storage/Projects/data/CelebAHDF5/celeba_aligned_cropped.hdf5','r')


# fvalid = h5py.File("celeba_aligned_cropped_valid_5cls.hdf5", "w")
# cls0_valid=valid['targets'][:,0]==1  
# cls1_valid=valid['targets'][:,1]==1  
# cls2_valid=valid['targets'][:,2]==1  
# cls3_valid=valid['targets'][:,3]==1  
# cls4_valid=valid['targets'][:,4]==1  
# icls_valid=np.where(np.logical_or.reduce((cls0_valid, cls1_valid, cls2_valid, cls3_valid, cls4_valid)))[0]

# fvalid.create_dataset("features", (icls_valid.shape[0],218,178,3), dtype='f')
# fvalid.create_dataset("targets", (icls_valid.shape[0],5), dtype='f')


# print('Generate valid features')
# for i in range(icls_valid.shape[0]):                                     
#    fvalid['features'][i]=valid['features'][icls_valid[i],:,:,:]
   
# print('Generate valid targets')
# for i in range(icls_valid.shape[0]):                                     
#    fvalid['targets'][i]=valid['targets'][icls_valid[i],0:5]  

# ftest = h5py.File("celeba_aligned_cropped_test_5cls.hdf5", "w")
# cls0_test=test['targets'][:,0]==1  
# cls1_test=test['targets'][:,1]==1  
# cls2_test=test['targets'][:,2]==1  
# cls3_test=test['targets'][:,3]==1  
# cls4_test=test['targets'][:,4]==1  
# icls_test=np.where(np.logical_or.reduce((cls0_test, cls1_test, cls2_test, cls3_test, cls4_test)))[0]

# ftest.create_dataset("features", (icls_test.shape[0],218,178,3), dtype='f')
# ftest.create_dataset("targets", (icls_test.shape[0],5), dtype='f')

# print('Generate test features')
# for i in range(icls_test.shape[0]):                                     
#    ftest['features'][i]=test['features'][icls_test[i],:,:,:]
       
# print('Generate test targets')
# for i in range(icls_test.shape[0]):                                     
#    ftest['targets'][i]=test['targets'][icls_test[i],0:5]


IPython.embed()