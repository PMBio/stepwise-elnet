import sys
sys.path.append('./..')
from CFG.settings import *
import include.data as DATA
from include.utils import matchIDs
from include.utils import smartDumpDictHdf5
from include.normalization import regressOut
from sklearn import cross_validation
from sklearn.linear_model import ElasticNetCV

import h5py
import pdb
import os
import scipy as sp
import scipy.stats as ST
import pandas as PD
import os
import itertools

def CondElNet(Y,X1,X2=None,opt1=None,opt2=None,cv=None,cv_opt=None,X1star=None,X2star=None):
    """
    Y:   Nx1 outcome
    X1:  NxF1 features 
    X2:  NxF2 features (if specified X2 is fitted on top of Y after regressing out X1)
    opts1: dictionary of opts for fitting (Y,X1) 
    opts2: dictionary of opts for fitting (Y,X2) 
    cv: sklearn cross validation 
    cv_opt: option for cross validation 
    """
    if cv is None:
        cv = cross_validation.KFold(**cv_opt)

    RV = {}

    # fit (Y,X1) 
    elnet_som = ElasticNetCV(cv=cv,**opt1)
    elnet_som.fit(X1, Y_train.ravel())
    RV['weights1'] = elnet_som.coef_
    if X1star is not None:
        RV['Ystar1'] = elnet_som.predict(X1star)

    # fit (Y,X2) 
    if X2 is not None:
        Y_train_r = Y_train.ravel()-elnet_som.predict(X1)
        elnet_germ = ElasticNetCV(cv=cv,**opt2)
        elnet_germ.fit(X2, Y_train_r)
        RV['weights2'] = elnet_germ.coef_
        if X2star is not None:
            RV['Ystar2'] = elnet_germ.predict(X2star)
            RV['Ystar'] = RV['Ystar1']+RV['Ystar2']

    return RV


if __name__=='__main__':

    # params
    opt = {}
    opt['l1_ratio'] = sp.arange(1,11)*.1
    opt['copy_X'] = False
    opt['fit_intercept'] = True
    opt['max_iter'] = 5000
    opt1 = opt
    opt2 = opt
    # alphas som
    upper = -.01
    lower = -2.
    inv = (upper - lower) / 25.
    opt1['alphas'] = 10.**sp.arange(lower, upper, inv)
    # alphas germ
    upper = -.001
    lower = -1.
    inv = (upper - lower) / 25.
    opt2['alphas'] = 10.**sp.arange(lower, upper, inv)

    # generate data
    N  = 500
    F1 = 100
    F2 = 100
    Y  = sp.randn(N,1)
    X1 = sp.rand(N,F1)
    X2 = sp.rand(N,F2)

    # for crossvalidation
    nfolds = 5
    cv_idx = 0

    # splitting in train and test 
    seed = 0
    cv=cross_validation.KFold(n=Y.shape[0],shuffle=True,n_folds=nfolds,random_state=seed)
    idx_train,idx_test = next(itertools.islice(cv,cv_idx,cv_idx+1))
    Y_train  = Y[idx_train];  Y_test  = Y[idx_test]
    X1_train = X1[idx_train]; X1_test = X1[idx_test]
    X2_train = X2[idx_train]; X2_test = X2[idx_test]

    cv_opt = {'n':Y_train.shape[0],'shuffle':True,'n_folds':nfolds,'random_state':0}

    res = CondElNet(Y_train,X1_train,X2=X2_train,opt1=opt1,opt2=opt2,cv_opt=cv_opt,X1star=X1_test,X2star=X2_test)

