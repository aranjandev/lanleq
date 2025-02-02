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
EVAL_STEP = 100
CV_COUNT = 1
CALLBACK_SAVE_ITER = EVAL_STEP

# set general training params
train_params = {
    'N_ROUND': 8000, # number of learners
}

jd = json.JSONDecoder()
settings = jd.decode(open('settings.json').read())
starttime_str = dt.datetime.strftime(dt.datetime.now(), '%d%m%y%H%M%S')

# some globals for saving best models
g_best_val_model_file = settings['TMP_OUTPUT'] + 'best-val-model-' + starttime_str + '.npy'
g_val_best_error = 1e5

# configure logger
logging.basicConfig(filename=settings['TMP_OUTPUT'] + 'run-' + starttime_str + '.log', level=logging.DEBUG, filemode='w')

# set xgb params
xgb_params = {'nthread': settings['XGB_NTHREAD'],
              'max_depth': 8,
              'eta': 0.001,
              'gamma': 0.1,
              'lambda': 0.1,
              'min_child_weight': 1,
              'objective': 'reg:linear',
              'eval_metric': 'mae',
              'subsample': 0.5,
              'colsample_bytree': 0.5,
              'silent': 1
              }

# save the best classifier so far
def callback_save_bestsofar(env):
    global g_val_best_error, g_best_val_model_file
    err = env.evaluation_result_list[0][1]
    if env.iteration % CALLBACK_SAVE_ITER == 0:
        if err < g_val_best_error:
            np.save(g_best_val_model_file, [env.model, env.iteration, err])
            g_val_best_error = err
            print('-- Saved best model w err: {:.2f} to {}'.format(g_val_best_error, g_best_val_model_file))

def run_train(settings, xgb_params, TRAINING_SAMPS):
    # set up the original training data
    ftrain = h5py.File(settings['TRAIN_FEATURES_X'], 'r')
    full_tr_count = ftrain[settings['DS_TRAIN_FEATURES']].shape[0]
    if TRAINING_SAMPS == 0:
        TRAINING_SAMPS = full_tr_count

    # reset global val error
    global g_val_best_error 
    g_val_best_error = 1e5

    saved_model_file = settings['SUBMISSION_FOLDER'] + 'model-' + dt.datetime.strftime(dt.datetime.now(), '%d%m%y%H%M%S') + '.npy'

    logging.debug('-- Training using settings: {}'.format(settings))
    logging.debug("-- XGBoost params: {}".format(xgb_params))
    logging.debug("-- General training params: {}".format(train_params))

    # Split full traing data to random TRAIN and EVAL set indices
    all_samp_ind = random.sample(range(full_tr_count), k=TRAINING_SAMPS)
    random.shuffle(all_samp_ind)
    train_ind_rand = np.sort(all_samp_ind[0:int((1-EVAL_FRAC)*TRAINING_SAMPS)])
    eval_ind_rand = np.sort(all_samp_ind[int((1-EVAL_FRAC)*TRAINING_SAMPS)+1:])

    if GENERATE_LIBSVM:
        logging.debug('-- Generating LIBSVM training data: train index = {} ({}), eval index = {} ({})'.format(train_ind_rand.shape, train_ind_rand, eval_ind_rand.shape, eval_ind_rand))
        allTrainX = ftrain[settings['DS_TRAIN_FEATURES']][train_ind_rand, 0:TRAINING_FEA_DIM,:].reshape(-1,TRAINING_FEA_DIM)
        allTrainY = np.load(settings['TRAIN_FEATURES_Y'])[train_ind_rand, :]
        logging.debug('-- processing train = ({},{})'.format(allTrainX.shape, allTrainY.shape))
        if EVAL_FRAC > 0:
            allEvalX = ftrain[settings['DS_TRAIN_FEATURES']][eval_ind_rand, 0:TRAINING_FEA_DIM,:].reshape(-1,TRAINING_FEA_DIM)
            allEvalY = np.load(settings['TRAIN_FEATURES_Y'])[eval_ind_rand, :]
            logging.debug('-- processing eval = ({},{})...'.format(allEvalX.shape, allEvalY.shape))
        import sklearn.datasets as skd
        starttime=time.time()
        skd.dump_svmlight_file(allTrainX, allTrainY.ravel(), settings['TRAINDATA_LIBSVM'])
        allTrainX = []
        allTrainY = []
        if EVAL_FRAC > 0:
            skd.dump_svmlight_file(allEvalX, allEvalY.ravel(), settings['EVALDATA_LIBSVM'])
            allEvalX = []
            allEvalY = []
        endtime=time.time()
        logging.debug('-- Dumped LIBSVM data file in {:.2f} min'.format((endtime-starttime)/60))

    if USE_LIBSVM:
        cache_train_data_path = settings['TRAINDATA_LIBSVM'] + '#' + settings['CACHE_TRAIN_LIBSVM']
        logging.debug('-- Loading cached TRAIN data from {} '.format(cache_train_data_path))
        dtrain = xgb.DMatrix(cache_train_data_path)
        if EVAL_FRAC > 0:
            cache_eval_data_path = settings['EVALDATA_LIBSVM'] + '#' + settings['CACHE_EVAL_LIBSVM']
            logging.debug('-- Loading cached EVAL data from {} '.format(cache_eval_data_path))
            deval = xgb.DMatrix(cache_eval_data_path)
    else:
        allTrainX = ftrain[settings['DS_TRAIN_FEATURES']][train_ind_rand, 0:TRAINING_FEA_DIM,:].reshape(-1,TRAINING_FEA_DIM)
        allTrainY = np.load(settings['TRAIN_FEATURES_Y'])[train_ind_rand, :]
        dtrain = xgb.DMatrix(allTrainX, allTrainY)
        logging.debug('-- dtrain loaded: {} x {}. Starting training...'.format(dtrain.num_row(), dtrain.num_col()))
        if EVAL_FRAC > 0:
            allEvalX = ftrain[settings['DS_TRAIN_FEATURES']][eval_ind_rand, 0:TRAINING_FEA_DIM,:].reshape(-1,TRAINING_FEA_DIM)
            allEvalY = np.load(settings['TRAIN_FEATURES_Y'])[eval_ind_rand, :]        
            deval = xgb.DMatrix(allEvalX, allEvalY)    
            logging.debug('-- deval loaded: {} x {}. Starting training...'.format(deval.num_row(), deval.num_col()))
    # training classifier
    starttime = time.time()
    if EVAL_FRAC > 0:
        watchlist = [(deval,'eval'),(dtrain,'train')]
        evals_result = {}
    else:
        watchlist = [(dtrain,'train')]
        evals_result = {}
    #best_classifier = xgb.train(xgb_params, dtrain, train_params['N_ROUND'], watchlist, evals_result=evals_result, verbose_eval=EVAL_STEP)
    best_classifier = xgb.train(xgb_params, dtrain, train_params['N_ROUND'], watchlist, evals_result=evals_result, verbose_eval=EVAL_STEP, callbacks=[callback_save_bestsofar])

    endtime = time.time()
    elapsed_sec = endtime - starttime
    logging.debug('-- XGB training completed in {0:.2f} sec ({1:.2f} hrs)'.format(elapsed_sec, elapsed_sec/3600))
    np.save(saved_model_file, [best_classifier, xgb_params, train_params, evals_result])
    logging.debug('-- Saved the full classifier and evals_result to {0}'.format(saved_model_file))

for cv_iter in range(CV_COUNT):
    print('Running iter {}'.format(cv_iter))
    run_train(settings=settings, xgb_params=xgb_params, TRAINING_SAMPS=TRAINING_SAMPS)