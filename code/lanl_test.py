import numpy as np
import json
import feautils
import time
import glob
import pandas as pd
import xgboost as xgb
import os
import argparse

def main(args):
    settings_file = './settings.json'
    PARALLEL_PROC_SAMP_COUNT = 390
    NJOBS_FEATURE = 20
    CALC_TEST_FEAS = False
    MODEL_NAME = args.model #'model-230519162220'
    IS_INTERMEDIATE_MODEL = args.intermediate

    jd = json.JSONDecoder()
    settings = jd.decode(open(settings_file).read())

    if CALC_TEST_FEAS:
        # read all the signals
        fnames = glob.glob(settings['TESTX'] + '*.csv')
        signal_list = []
        print('-- Reading signals from {} files in {}'.format(len(fnames), settings['TESTX']))
        for f in fnames:
            df = pd.read_csv(f)
            sig = np.asarray(df.iloc[:,0]).reshape(-1,1)
            signal_list.append(sig.reshape(-1,1))
            print('--- Done {} (list len = {})'.format(f, len(signal_list)))

        print('-- Parallel feature calculation for {} signals'.format(len(signal_list)))
        TestFeaMat = feautils.parallel_feature_calc(signal_list, n_jobs=NJOBS_FEATURE)
        np.save(settings['TEST_FEATURES'], TestFeaMat)
        print('-- Saved test features of size {} to {}'.format(TestFeaMat.shape, settings['TEST_FEATURES']))

    # Run classifier on the test data
    # -----------------------
    # load stored model
    model_file = settings['SUBMISSION_FOLDER'] + MODEL_NAME + '.npy'
    print('-- Loading model from {} (Intermediate model = {})'.format(model_file,IS_INTERMEDIATE_MODEL))
    model_ld = np.load(model_file)
    xgb_model = model_ld[0]
    if IS_INTERMEDIATE_MODEL:
        ntree_limit = model_ld[1]
    else:
        ntree_limit = model_ld[2]['N_ROUNDS']
    # create dict for results
    test_X = np.load(settings['TEST_FEATURES'])
    print('-- Loaded test_X of shape: {}'.format(test_X.shape))
    # loop over and test
    print('-- Predicting on all test data with {} trees...'.format(ntree_limit))
    dtest = xgb.DMatrix(test_X)
    predY = xgb_model.predict(dtest, ntree_limit=ntree_limit)

    # preparing submission
    submission_file = settings['SUBMISSION_FOLDER'] + 'submission-' + MODEL_NAME + '.csv'
    test_fnames = glob.glob(settings['TESTX'] + '*.csv')
    seg_id = []
    for fname in test_fnames:
        seg_id.append(os.path.basename(fname).split('.')[0])

    submission_df = pd.DataFrame({'seg_id': seg_id,'time_to_failure': list(predY)})
    submission_df.to_csv(submission_file, index=False)
    print('-- Save submission to {}'.format(submission_file))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LANL prediction")
    parser.add_argument('--model', help="Model file prefix")
    parser.add_argument('--intermediate', action='store_true',
                        help="This is an intermediate model")
    main(parser.parse_args())
