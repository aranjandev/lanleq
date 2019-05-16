from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import json
import h5py
import feautils
import time
import glob

settings_file = './settings.json'
PARALLEL_PROC_SAMP_COUNT = 390
NJOBS_FEATURE = 20

jd = json.JSONDecoder()
settings = jd.decode(open(settings_file).read())

# read all the signals
fnames = glob.glob(settings['TESTX'] + '*.csv')
signal_list = []
print('-- Reading signals from {} files in {}'.format(len(fnames), settings['TESTX']))
for f in fnames:
    df = pd.read_csv(f)
    sig = np.asarray(df.iloc[:,0]).reshape(-1,1)
    signal_list.append(sig.reshape(-1,1))

print('-- Parallel feature calculation for {} signals'.format(len(signal_list)))
TestFeaMat = feautils.parallel_feature_calc(signal_list, n_jobs=NJOBS_FEATURE)
