import numpy as np
import feautils
import xgboost as xgb
import h5py
import datetime as dt
import random
import json
import argparse
import time
import logging

# random seed for repeatability
random.seed(0)
np.random.seed(0)
TRAINING_SAMPS = 0
TRAINING_FEA_DIM = feautils.FEA_DIM
GENERATE_LIBSVM = False
USE_LIBSVM = True
EVAL_FRAC = 0.2
EVAL_STEP = 50

# set general training params
train_params = {
    'N_ROUND': 5000, # number of learners
}

jd = json.JSONDecoder()
settings = jd.decode(open('settings.json').read())

# configure logger
logging.basicConfig(filename=settings['TMP_OUTPUT']+'run.log',level=logging.DEBUG, filemode='w')

# set xgb params
xgb_params = {'nthread': settings['XGB_NTHREAD'],
              'max_depth': 6,
              'eta': 0.01,
              'gamma': 0.1,
              'lambda': 0.1,
              'min_child_weight': 1,
              'objective': 'reg:linear',
              'eval_metric': 'mae',
              'subsample': 0.5,
              'colsample_bytree': 0.5,
              'silent': 0
              }

saved_model_file = settings['SUBMISSION_FOLDER'] + 'model-' + dt.datetime.strftime(dt.datetime.now(), '%d%m%y%H%M%S') + '.npy'

logging.debug('-- Training using settings: {}'.format(settings))
logging.debug("-- XGBoost params: {}".format(xgb_params))
logging.debug("-- General training params: {}".format(train_params))

ftrain = h5py.File(settings['TRAIN_FEATURES_X'], 'r')
full_tr_count = ftrain[settings['DS_TRAIN_FEATURES']].shape[0]
if TRAINING_SAMPS == 0:
    TRAINING_SAMPS = full_tr_count

# Split full traing data to random TRAIN and EVAL set indices
all_samp_ind = np.arange(0, TRAINING_SAMPS)
random.shuffle(all_samp_ind)
train_ind_rand = np.sort(all_samp_ind[0:int((1-EVAL_FRAC)*TRAINING_SAMPS)])
eval_ind_rand = np.sort(all_samp_ind[int((1-EVAL_FRAC)*TRAINING_SAMPS)+1:])

if GENERATE_LIBSVM:
    logging.debug('-- Generating LIBSVM training data: train index = {} ({}), eval index = {} ({})'.format(train_ind_rand.shape, train_ind_rand, eval_ind_rand.shape, eval_ind_rand))
    allTrainX = ftrain[settings['DS_TRAIN_FEATURES']][train_ind_rand, 0:TRAINING_FEA_DIM,:].reshape(-1,TRAINING_FEA_DIM)
    allTrainY = np.load(settings['TRAIN_FEATURES_Y'])[train_ind_rand, :]
    allEvalX = ftrain[settings['DS_TRAIN_FEATURES']][eval_ind_rand, 0:TRAINING_FEA_DIM,:].reshape(-1,TRAINING_FEA_DIM)
    allEvalY = np.load(settings['TRAIN_FEATURES_Y'])[eval_ind_rand, :]
    logging.debug('-- processing train = ({},{}), eval = ({},{})...'.format(allTrainX.shape, allTrainY.shape, allEvalX.shape, allEvalY.shape))
    import sklearn.datasets as skd
    starttime=time.time()
    skd.dump_svmlight_file(allTrainX, allTrainY.ravel(), settings['TRAINDATA_LIBSVM'])
    skd.dump_svmlight_file(allEvalX, allEvalY.ravel(), settings['EVALDATA_LIBSVM'])
    allTrainX = []
    allTrainY = []
    allEvalX = []
    allEvalY = []
    endtime=time.time()
    logging.debug('-- Dumped LIBSVM data file in {:.2f} min'.format((endtime-starttime)/60))

if USE_LIBSVM:
    cache_train_data_path = settings['TRAINDATA_LIBSVM'] + '#' + settings['CACHE_TRAIN_LIBSVM']
    logging.debug('-- Loading cached TRAIN data from {} '.format(cache_train_data_path))
    dtrain = xgb.DMatrix(cache_train_data_path)
    cache_eval_data_path = settings['EVALDATA_LIBSVM'] + '#' + settings['CACHE_EVAL_LIBSVM']
    logging.debug('-- Loading cached EVAL data from {} '.format(cache_eval_data_path))
    deval = xgb.DMatrix(cache_eval_data_path)
else:
    allTrainX = ftrain[settings['DS_TRAIN_FEATURES']][train_ind_rand, 0:TRAINING_FEA_DIM,:].reshape(-1,TRAINING_FEA_DIM)
    allTrainY = np.load(settings['TRAIN_FEATURES_Y'])[train_ind_rand, :]
    allEvalX = ftrain[settings['DS_TRAIN_FEATURES']][eval_ind_rand, 0:TRAINING_FEA_DIM,:].reshape(-1,TRAINING_FEA_DIM)
    allTrainY = np.load(settings['TRAIN_FEATURES_Y'])[eval_ind_rand, :]
    dtrain = xgb.DMatrix(allTrainX, allTrainY)
    deval = xgb.DMatrix(allEvalX, allEvalY)

logging.debug('-- dtrain loaded: {} x {}. Starting training...'.format(dtrain.num_row(), dtrain.num_col()))
logging.debug('-- deval loaded: {} x {}. Starting training...'.format(deval.num_row(), deval.num_col()))
# training classifier
starttime = time.time()
watchlist = [(deval,'eval'),(dtrain,'train')]
evals_result = {}
best_classifier = xgb.train(xgb_params, dtrain, train_params['N_ROUND'], watchlist, evals_result=evals_result, verbose_eval=EVAL_STEP)
dtrain = None
deval = None
endtime = time.time()
elapsed_sec = endtime - starttime
logging.debug('-- XGB training completed in {0:.2f} sec ({1:.2f} hrs)'.format(elapsed_sec, elapsed_sec/3600))
np.save(saved_model_file, [best_classifier, xgb_params, train_params, evals_result])
logging.debug('-- Saved the full classifier and evals_result to {0}'.format(saved_model_file))
