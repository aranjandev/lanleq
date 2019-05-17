import numpy as np
import feautils
import xgboost as xgb
import h5py
import datetime as dt
import random
import json
import argparse
import time

# random seed for repeatability
random.seed(0)
np.random.seed(0)
TRAINING_SAMPS = 5000

# set general training params
train_params = {
    'N_ROUND': 200, # number of learners
}

jd = json.JSONDecoder()
settings = jd.decode(open('settings.json').read())

# set xgb params
xgb_params = {'nthread': settings['XGB_NTHREAD'],
              'max_depth': 2,
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

###########
# Training
###########
#def main(args):
# if args.simple:
#     # train a simplified classifier
#     train_params['N_ROUND'] = 100
#     xgb_params['eta'] = 0.1

saved_model_file = './localdata/model-{}.npy'.format(dt.datetime.strftime(dt.datetime.now(), '%d%m%y%H%M%S'))

print('-- Training using settings: {}'.format(settings))
print("-- XGBoost params: {}".format(xgb_params))
print("-- General training params: {}".format(train_params))

# load precalculated training features
ftrain = h5py.File(settings['TRAIN_FEATURES_X'], 'r')
full_tr_count = ftrain[settings['DS_TRAIN_FEATURES']].shape[0]

if TRAINING_SAMPS == 0:
    TRAINING_SAMPS = full_tr_count

allX = ftrain[settings['DS_TRAIN_FEATURES']][0:TRAINING_SAMPS,:,:].reshape(-1,feautils.FEA_DIM)
allY = np.load(settings['TRAIN_FEATURES_Y'])[0:TRAINING_SAMPS, :]

dtrain = xgb.DMatrix(allX, allY)
# training classifier
print('-- Training starting with: X={}, Y={}'.format(allX.shape, allY.shape))
starttime = time.time()
#best_classifier = xgb.train(params=xgb_params, dtrain=dtrain, num_boost_round=train_params['N_ROUND'])
best_classifier = xgb.cv(params=xgb_params, dtrain=dtrain, num_boost_round=train_params['N_ROUND'], nfold=3, seed=0)
dtrain = None
endtime = time.time()
elapsed_sec = endtime - starttime
print('-- XGB training completed in {0:.2f} sec ({1:.2f} hrs)'.format(elapsed_sec, elapsed_sec/3600))
np.save(saved_model_file, [best_classifier, xgb_params])
print('-- Saved the full classifier to {0}'.format(saved_model_file))

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="LANL XGB training")
#     parser.add_argument('--simple', action='store_true',
#                         help="Simplified model training for fast execution")
#     main(parser.parse_args())    