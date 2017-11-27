import IPython
import h5py
import numpy as np
from keras.utils.io_utils import HDF5Matrix
import time

homepath = '/mnt/Storage'

train=h5py.File(homepath+'/Projects/data/CelebAHDF5/celeba_aligned_cropped_train_5cls.hdf5','r')

cls4_1=train['targets'][:,4]==1  
icls4=np.where(cls4_1)[0]
cnt4=icls4.shape[0]

start = time.time()

#sample 4
icls=icls4

#sample 2
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
icls=np.sort(icls)
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
	icls=np.sort(icls)

end = time.time()
print(end - start)

IPython.embed()


# ------------------------------------------------------------------------------------------------------------------
# ftrain = h5py.File("celeba_aligned_cropped_train_5cls_bootstrap2.hdf5", "w")  
# cls2_train=train['targets'][:,2]==1  
# cls4_train=train['targets'][:,4]==0  
# icls_train=np.where(np.logical_and.reduce((cls2_train, cls4_train)))[0]

# ftrain.create_dataset("features", (icls_train.shape[0],218,178,3), dtype='f')
# ftrain.create_dataset("targets", (icls_train.shape[0],5), dtype='f')


# print('Generate train features 2')
# for i in range(icls_train.shape[0]):                                     
#    ftrain['features'][i]=train['features'][icls_train[i],:,:,:]
   
# print('Generate train targets 2')
# for i in range(icls_train.shape[0]):                                     
#    ftrain['targets'][i]=train['targets'][icls_train[i],0:5]  

# ftrain.close()
# # ------------------------------------------------------------------------------------------------------------------
# ftrain = h5py.File("celeba_aligned_cropped_train_5cls_bootstrap1.hdf5", "w")
# cls1_train=train['targets'][:,1]==1  
# cls2_train=train['targets'][:,2]==0  
# cls4_train=train['targets'][:,4]==0  
# icls_train=np.where(np.logical_and.reduce((cls1_train, cls2_train, cls4_train)))[0]

# ftrain.create_dataset("features", (icls_train.shape[0],218,178,3), dtype='f')
# ftrain.create_dataset("targets", (icls_train.shape[0],5), dtype='f')


# print('Generate train features 1')
# for i in range(icls_train.shape[0]):                                     
#    ftrain['features'][i]=train['features'][icls_train[i],:,:,:]
   
# print('Generate train targets 1')
# for i in range(icls_train.shape[0]):                                     
#    ftrain['targets'][i]=train['targets'][icls_train[i],0:5]  

# ftrain.close()
# # ------------------------------------------------------------------------------------------------------------------
# ftrain = h5py.File("celeba_aligned_cropped_train_5cls_bootstrap3.hdf5", "w")
# cls1_train=train['targets'][:,1]==0  
# cls2_train=train['targets'][:,2]==0  
# cls3_train=train['targets'][:,3]==1  
# cls4_train=train['targets'][:,4]==0  
# icls_train=np.where(np.logical_and.reduce((cls1_train, cls2_train, cls3_train, cls4_train)))[0]

# ftrain.create_dataset("features", (icls_train.shape[0],218,178,3), dtype='f')
# ftrain.create_dataset("targets", (icls_train.shape[0],5), dtype='f')


# print('Generate train features 3')
# for i in range(icls_train.shape[0]):                                     
#    ftrain['features'][i]=train['features'][icls_train[i],:,:,:]
   
# print('Generate train targets 3')
# for i in range(icls_train.shape[0]):                                     
#    ftrain['targets'][i]=train['targets'][icls_train[i],0:5]  

# ftrain.close()
# # ------------------------------------------------------------------------------------------------------------------
# ftrain = h5py.File("celeba_aligned_cropped_train_5cls_bootstrap0.hdf5", "w")
# cls0_train=train['targets'][:,0]==1  
# cls1_train=train['targets'][:,1]==0  
# cls2_train=train['targets'][:,2]==0  
# cls3_train=train['targets'][:,3]==0  
# cls4_train=train['targets'][:,4]==0  
# icls_train=np.where(np.logical_and.reduce((cls0_train, cls1_train, cls2_train, cls3_train, cls4_train)))[0]

# ftrain.create_dataset("features", (icls_train.shape[0],218,178,3), dtype='f')
# ftrain.create_dataset("targets", (icls_train.shape[0],5), dtype='f')


# print('Generate train features 0')
# for i in range(icls_train.shape[0]):                                     
#    ftrain['features'][i]=train['features'][icls_train[i],:,:,:]
   
# print('Generate train targets 0')
# for i in range(icls_train.shape[0]):                                     
#    ftrain['targets'][i]=train['targets'][icls_train[i],0:5]  

# ftrain.close()

# # IPython.embed()