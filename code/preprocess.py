import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
import json
from collections import deque
import h5py
import feautils
import time

TOTAL_TRAIN_LEN = 629145481
settings_file = './settings.json'
TEST_LEN = 150000
STEP_SZ = 5000
PARALLEL_PROC_SAMP_COUNT = 390
TRAIN_SAMPLE_COUNT = int((TOTAL_TRAIN_LEN - TEST_LEN)/STEP_SZ) # = 125970=390*N
NJOBS_FEATURE = 20

jd = json.JSONDecoder()
settings = jd.decode(open(settings_file).read())

chunk_reader = pd.read_csv(settings['TRAIN_CSV'], chunksize=STEP_SZ)
curr_deque = deque()
max_deq_len = int(TEST_LEN/STEP_SZ)

# create hdf5 dataset
trainY = np.zeros((TRAIN_SAMPLE_COUNT,1))
# create output data set
print("-- Creating output feature dataset in {}".format(settings['DS_TRAIN_FEATURES']))
f_out = h5py.File(settings['TRAIN_FEATURES_X'], 'w')
f_out.create_dataset(settings['DS_TRAIN_FEATURES'], (TRAIN_SAMPLE_COUNT, feautils.FEA_DIM, 1))
features_out = f_out[settings['DS_TRAIN_FEATURES']]

st = time.time()
print('-- Starting segment feature extraction for {} samples...'.format(TRAIN_SAMPLE_COUNT))
train_sigseg_index = 0
train_fea_index = 0
start_ind = 0
signal_list = []
for chunk in chunk_reader:
    ac_data = np.asarray(chunk.iloc[:,0])
    curr_deque.append(ac_data)
    if len(curr_deque) == max_deq_len:
        # signal segment calculation
        x = np.asarray(curr_deque).flatten().reshape(-1,1)
        y = chunk.iloc[-1,1]
        signal_list.append(x)
        trainY[train_sigseg_index] = y
        curr_deque.popleft()
        #print('done chunk: {}'.format(train_sigseg_index))
        if len(signal_list) == PARALLEL_PROC_SAMP_COUNT:
            print('-- Buffered {} signals. Calculating feas in parallel...'.format(len(signal_list)))
            # calculate all features in parallel
            feaMat = feautils.parallel_feature_calc(signal_list, n_jobs=NJOBS_FEATURE)
            # reset the buffer
            signal_list = []
            # store features into hdf5
            features_out[train_fea_index : train_fea_index + PARALLEL_PROC_SAMP_COUNT, :, :] = feaMat.reshape(PARALLEL_PROC_SAMP_COUNT, -1, 1)
            print('-- processed HDF5 idx: {}-{}'.format(train_fea_index, train_fea_index + PARALLEL_PROC_SAMP_COUNT))
            train_fea_index += PARALLEL_PROC_SAMP_COUNT

        train_sigseg_index += 1
    if train_sigseg_index == TRAIN_SAMPLE_COUNT:
        break

np.save(settings['TRAIN_FEATURES_Y'], trainY)
f_out.close()
et = time.time()
print("-- Finished {} in {:.2f} sec ({:.2f} sec per sample)".format(TRAIN_SAMPLE_COUNT, et-st, (et-st)/TRAIN_SAMPLE_COUNT))