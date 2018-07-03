import sys
sys.path.append('./..')
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
from optparse import OptionParser

def CondElNet(Y_train, X1 , X1star, elnet_som, X2, X2star, elnet_germ):
    """
    Parameters
    ----------
    Y:   (`N`, `1`) ndarray
        Outcome
    X1:  (`N`, `F1`) ndarray
        First set of features (train data)
    X2:  (`N`, `F2`) ndarray
        Second set of features (train data)
    X1star:  (`Ns`, `F1`) ndarray
        First set of features (test data)
    X2star:  (`Ns`, `F2`) ndarray
        Second set of features (test data)
    elnet_som: sklearn model
        Elastic net for somatic model
    elnet_germ: sklearn model
        Elastic net for germline model

    Returns
    -------
    Dictionaty with weights and out-of-sample predictions
    """
    RV = {}

    # fit (Y,X1)
    elnet_som.fit(X1, Y_train.ravel())
    RV['weights1'] = elnet_som.coef_
    RV['Ystar1'] = elnet_som.predict(X1star)

    # fit (Y,X2)
    Y_train_r = Y_train.ravel()-elnet_som.predict(X1)
    elnet_germ.fit(X2, Y_train_r)
    RV['weights2'] = elnet_germ.coef_
    RV['Ystar2'] = elnet_germ.predict(X2star)
    RV['Ystar'] = RV['Ystar1']+RV['Ystar2']

    return RV

def generate_data():
    # generate data
    N  = 500
    F1 = 100
    F2 = 100
    Y  = sp.randn(N,1)
    X1 = sp.rand(N,F1)
    X2 = sp.rand(N,F2)
    return Y, X1, X2

if __name__=='__main__':

    parser = OptionParser()
    parser.add_option("--seed", dest='seed', type=int, default=0)
    parser.add_option("--nfolds", dest='nfolds', type=int, default=5)
    parser.add_option("--fold_i", dest='fold_i', type=int, default=0)
    parser.add_option("--outfile", dest='outfile', type=str, default="out.h5")
    (opt, args) = parser.parse_args()
    opt_dict = vars(opt)

    # fine params of the two elastic net
    opt_elnet = {}
    opt_elnet['l1_ratio'] = sp.arange(1,11)*.1
    opt_elnet['copy_X'] = False
    opt_elnet['fit_intercept'] = True
    opt_elnet['max_iter'] = 5000
    opt_elnet1 = opt_elnet
    opt_elnet2 = opt_elnet
    # alphas som
    upper = -.01
    lower = -2.
    inv = (upper - lower) / 25.
    opt_elnet1['alphas'] = 10.**sp.arange(lower, upper, inv)
    # alphas germ
    upper = -.001
    lower = -1.
    inv = (upper - lower) / 25.
    opt_elnet2['alphas'] = 10.**sp.arange(lower, upper, inv)

    # load/generate data
    Y, X1, X2 = generate_data()

    # splitting in train and test
    cv = cross_validation.KFold(n=Y.shape[0],shuffle=True,n_folds=opt.nfolds,random_state=opt.seed)
    idx_train, idx_test = next(itertools.islice(cv,opt.fold_i,opt.fold_i+1))
    Y_train  = Y[idx_train];  Y_test  = Y[idx_test]
    X1_train = X1[idx_train]; X1_test = X1[idx_test]
    X2_train = X2[idx_train]; X2_test = X2[idx_test]

    # define inner crossvaludation and conditional elnet
    cv_opt = {'n':Y_train.shape[0],'shuffle':True,'n_folds':opt.nfolds,'random_state':0}
    cv_inner = cross_validation.KFold(**cv_opt)
    elnet_som = ElasticNetCV(cv=cv_inner,**opt_elnet1)
    elnet_germ = ElasticNetCV(cv=cv_inner,**opt_elnet2)

    # run model
    res = CondElNet(Y_train,X1_train,X1_test,elnet_som,X2_train,X2_test,elnet_germ)
    res['idxs'] = idx_test
    res['Y'] = Y_test

    f = h5py.File(opt.outfile, 'w')
    for key in res.keys():
        f.create_dataset(key, data=res[key])
    f.close()
