import h5py
from keras.utils.io_utils import HDF5Matrix
import IPython
import numpy as np
def normalize_pixel(data):
    return data/255-.5


y4_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_train_5cls_bootstrap4.hdf5', 'targets')
y2_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_train_5cls_bootstrap2.hdf5', 'targets')
y1_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_train_5cls_bootstrap1.hdf5', 'targets')
y3_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_train_5cls_bootstrap3.hdf5', 'targets')
y0_data = HDF5Matrix('/home/ubuntuone/Projects/data/CelebAHDF5/celeba_aligned_cropped_train_5cls_bootstrap0.hdf5', 'targets')

y_data = y4_data[:]

cnt = np.where(y_data[:,2]==1)[0].shape[0]
# sample = np.random.random_integer(0,)

# y_data = np.concatenate((y_data, y2_data[sample]))


IPython.embed()


