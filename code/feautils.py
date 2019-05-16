import argparse
from numba import jit
import pandas as pd
from matplotlib import pyplot as plt
import time
import numpy as np
import time
import datetime
from joblib import Parallel, delayed
import json
import h5py

PHASE = 1
WORDLEN = 8
RADIICOUNT = 8
RADII = np.multiply(np.power(2,np.arange(0,RADIICOUNT)), WORDLEN)
DELTA = np.asarray([0,4,8])
ALL_WIN_BOUNDS = np.asarray([[0.0,1.0], [0.0,0.25], [0.25,0.5], [0.5,0.75], [0.75,1.0]])
FEA_DIM = PHASE * 2**WORDLEN * RADIICOUNT * DELTA.shape[0] * ALL_WIN_BOUNDS.shape[0]

def default_settings(setting_file='./settings.json'):
    jd = json.JSONDecoder()
    settings = jd.decode(open(setting_file).read())
    return settings

def default_feaparam():
    feaparam = {'medfilt_win': 101}
    return feaparam

# from feature index to feature generation description
def feature_detail(f_index):
    # word, radius, delta, win_bound, phase
    TOTAL_WORDS = 2**WORDLEN
    phase = int(np.floor(f_index/(ALL_WIN_BOUNDS.shape[0] * TOTAL_WORDS * RADIICOUNT * DELTA.size)))
    leftovers = f_index - (phase * ALL_WIN_BOUNDS.shape[0] * TOTAL_WORDS * RADIICOUNT * DELTA.size)
    win_bound = int(np.floor(leftovers/(TOTAL_WORDS * RADIICOUNT * DELTA.size)))
    leftovers = leftovers - (win_bound * TOTAL_WORDS * RADIICOUNT * DELTA.size) 
    delta = int(np.floor(leftovers/(TOTAL_WORDS * RADIICOUNT)))
    leftovers = leftovers - (delta * TOTAL_WORDS * RADIICOUNT)
    radius = int(np.floor(leftovers/TOTAL_WORDS))
    leftovers = leftovers - (radius * TOTAL_WORDS)
    word = int(leftovers)
    return {'phase': phase, 'win_bound': ALL_WIN_BOUNDS[win_bound], 'delta': DELTA[delta], 'radius': RADII[radius], 'word': word}

# pyramidal lbp computation with pyramid of windows
# radii must be multiple of word
# pass a column vector
# for fast optimization: remove all params other than vector arg.
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def pyrlbp_multiscale(inputsig):
    wordlen = WORDLEN
    radiicount = RADIICOUNT
    radii = RADII #np.multiply(np.power(2,np.arange(0,radiicount)), wordlen)
    delta = DELTA # [0,4,8]
    all_win_bounds = ALL_WIN_BOUNDS #ALL_WIN_BOUNDS #[[0.0,1.0], [0.0,0.25], [0.25,0.5], [0.5,0.75], [0.75,1.0]]
    powers = np.zeros((wordlen,1))
    samp_count = int(wordlen/2)
    pyrlbp_all = np.zeros((0), dtype=np.float64)
    for i in range(wordlen):
        powers[i] = 2 ** i
    for w_ind in np.arange(0, all_win_bounds.shape[0]):
        win_bound = all_win_bounds[w_ind,:]
        lbp_mult_delta = np.zeros((0), dtype=np.float64)
        win_start = round(win_bound[0] * float(inputsig.shape[0]))
        win_end = round(win_bound[1] * float(inputsig.shape[0]))
        if win_start < 0 or win_end > inputsig.shape[0] or win_end <= win_start:
            continue;
        vector = inputsig[win_start:win_end]
        for d in delta:
            lbp_ms = np.zeros((0), dtype=np.float64)
            for r in radii:
                lbp = np.zeros((2**wordlen), dtype=np.float64)
                factor = int(r/samp_count)
                for i in range(r, vector.shape[0]-r):
                    # center element to compare
                    x = vector[i]
                    # calculate before i
                    full_buff = vector[i-r:i]
                    # select word elements from this buffer
                    sel_buff = full_buff[0::factor]
                    bin_rep_before = np.greater_equal(sel_buff, x + d).astype(np.float64).reshape(1,-1)
                    # calculate after i
                    full_buff = vector[i+1:i+r+1]
                    # select word elements from this buffer
                    sel_buff = full_buff[0::factor]
                    bin_rep_after = np.greater_equal(sel_buff, x).astype(np.float64).reshape(1,-1)
                    # convert to decimal number
                    full_bin_rep = np.hstack((bin_rep_before, bin_rep_after))
                    int_word = full_bin_rep.dot(powers).ravel()[0]
                    hist_index = int(int_word)
                    lbp[hist_index] = lbp[hist_index] + 1.0
                lbp_ms = np.hstack((lbp_ms, lbp))
            lbp_mult_delta = np.hstack((lbp_mult_delta, lbp_ms))
        pyrlbp_all = np.hstack((pyrlbp_all, lbp_mult_delta))
    return pyrlbp_all

# Simplified fast feature
# radii must be multiple of word
# pass a column vector
# for fast optimization: remove all params other than vector arg.
@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit
def pyrlbp_multiscale_simple(inputsig):
    wordlen = WORDLEN
    radiicount = RADIICOUNT
    radii = RADII #np.multiply(np.power(2,np.arange(0,radiicount)), wordlen)
    delta = [DELTA[0]]
    all_win_bounds = ALL_WIN_BOUNDS[0,:].reshape(1,-1)
    powers = np.zeros((wordlen,1))
    samp_count = int(wordlen/2)
    pyrlbp_all = np.zeros((0), dtype=np.float64)
    for i in range(wordlen):
        powers[i] = 2 ** i    
    for w_ind in np.arange(0, all_win_bounds.shape[0]):
        win_bound = all_win_bounds[w_ind,:]
        lbp_mult_delta = np.zeros((0), dtype=np.float64)
        win_start = round(win_bound[0] * float(inputsig.shape[0]))
        win_end = round(win_bound[1] * float(inputsig.shape[0]))
        if win_start < 0 or win_end > inputsig.shape[0] or win_end <= win_start:
            continue;
        vector = inputsig[win_start:win_end]
        for d in delta:
            lbp_ms = np.zeros((0), dtype=np.float64)
            for r in radii:
                lbp = np.zeros((2**wordlen), dtype=np.float64)
                factor = int(r/samp_count)
                for i in range(r, vector.shape[0]-r):
                    # center element to compare
                    x = vector[i]
                    # calculate before i
                    full_buff = vector[i-r:i]
                    # select word elements from this buffer
                    sel_buff = full_buff[0::factor]
                    bin_rep_before = np.greater_equal(sel_buff, x + d).astype(np.float64).reshape(1,-1)
                    # calculate after i
                    full_buff = vector[i+1:i+r+1]
                    # select word elements from this buffer
                    sel_buff = full_buff[0::factor]
                    bin_rep_after = np.greater_equal(sel_buff, x).astype(np.float64).reshape(1,-1)
                    # convert to decimal number
                    full_bin_rep = np.hstack((bin_rep_before, bin_rep_after))
                    int_word = full_bin_rep.dot(powers).ravel()[0]
                    hist_index = int(int_word)
                    lbp[hist_index] = lbp[hist_index] + 1.0
                lbp_ms = np.hstack((lbp_ms, lbp))
            lbp_mult_delta = np.hstack((lbp_mult_delta, lbp_ms))
        pyrlbp_all = np.hstack((pyrlbp_all, lbp_mult_delta))
    return pyrlbp_all

# create output HDF5 dataset and populates in parallel
# extract feature using function: extract_fea_mat
def parallel_feature_calc(signal_list, n_jobs=8):
    print("-- Calculating features for mat: {}".format(len(signal_list)))
    st = time.time()
    allProcessedOut = Parallel(n_jobs=n_jobs, verbose=8)(delayed(pyrlbp_multiscale)(sig) for sig in signal_list)
    outMat = np.asarray(allProcessedOut)
    et = time.time()
    print('- Total time calculating features {0} = {1:.2f} sec'.format(outMat.shape, et-st))
    return outMat

def main(args):
    settings = default_settings()
    feaParams = default_feaparam()
    if args.simple:
        fea_proc_fun = pyrlbp_multiscale_simple
    else:
        fea_proc_fun = pyrlbp_multiscale
    print('-- Processing feature using: {0}'.format(fea_proc_fun))
    if args.train:
        parallel_feature_hdf5_cols(settings['TRAINX'], settings['TRAINY'], fea_proc_fun=fea_proc_fun, bSave=True, savepath=settings['TRAIN_FEATURES'], n_jobs=settings['FEATURE_NJOBS'])
    if args.test:
        parallel_feature_hdf5_cols(settings['TESTX'], settings['TEST_META'], fea_proc_fun=fea_proc_fun, bSave=True, savepath=settings['TEST_FEATURES'], n_jobs=settings['FEATURE_NJOBS'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="VSB feature extraction")
    parser.add_argument('--train', action='store_true',
                        help="Extract training features")
    parser.add_argument('--test', action='store_true',
                        help="Extract test features")
    parser.add_argument('--simple', action='store_true',
                        help="Simplified feature extraction for fast execution")
    main(parser.parse_args())
