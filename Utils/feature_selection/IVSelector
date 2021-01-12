import re
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from termcolor import cprint

import lightgbm as lgb
from joblib import Parallel, delayed
import time

import toad
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.base import TransformerMixin, BaseEstimator, ClassifierMixin
from sklearn.metrics import roc_auc_score, roc_curve


class Metrics:
    @classmethod
    def get_auc(cls, ytrue, yprob, **kwargs):
        auc = roc_auc_score(ytrue, yprob)

        if kwargs.get('symmetry', False) is True:
            if auc < 0.5:
                auc = 1 - auc
        return auc

    @classmethod
    def get_ks(cls, ytrue, yprob):
        fpr, tpr, thr = roc_curve(ytrue, yprob)
        ks = max(abs(tpr - fpr))
        return ks

    @classmethod
    def get_gini(cls, ytrue, yprob, **kwargs):
        auc = cls.get_auc(ytrue, yprob, **kwargs)
        gini = 2 * auc - 1

        return gini

    @classmethod
    def get_stat(cls, df_label, df_feature):
        var = df_feature.name
        df_data = pd.DataFrame({'val': df_feature, 'label': df_label})

# statistics of total count, total ratio, bad count, bad rate

        df_stat = df_data.groupby('val').agg(total=('label', 'count'),
                                             bad=('label', 'sum'),
                                             bad_rate=('label', 'mean'))
        df_stat['var'] = var
        df_stat['good'] = df_stat['total'] - df_stat['bad']
        df_stat['total_ratio'] = df_stat['total'] / df_stat['total'].sum()
        df_stat['good_density'] = df_stat['good'] / df_stat['good'].sum()
        df_stat['bad_density'] = df_stat['bad'] / df_stat['bad'].sum()

        eps = np.finfo(np.float32).eps
        df_stat.loc[:, 'iv'] = (df_stat['bad_density'] - df_stat['good_density']) * \
                               np.log((df_stat['bad_density'] + eps) / (df_stat['good_density'] + eps))

        cols = ['var', 'total', 'total_ratio', 'bad', 'bad_rate', 'iv', 'val']
        df_stat = df_stat.reset_index()[cols].set_index('var')
        return df_stat

    @classmethod
    def get_iv(cls, df_label, df_feature):
        df_stat = cls.get_stat(df_label, df_feature)
        return df_stat['iv'].sum()


class Logit(BaseEstimator, ClassifierMixin):
    def __init__(self):
        self.model = None
        self.detail = None

    def fit(self, df_xtrain, df_ytrain, **kwargs):
        df_xtrain_const = sm.add_constant(df_xtrain)

# model training. default using newton method, if fail use bfgs method

        try:
            self.model = sm.Logit(df_ytrain, df_xtrain_const).fit(method='newton', maxiter=100)
        except:
            cprint("warning:  exist strong correlated features, "
                   "got singular matrix in linear model, retry bfgs method instead.",
                   'red')
            self.model = sm.Logit(df_ytrain, df_xtrain_const).fit(method='bfgs', maxiter=100)

# prepare model result

        self.detail = pd.DataFrame({'var': df_xtrain_const.columns.tolist(),
                                    'coef': self.model.params,
                                    'std_err': [round(v, 3) for v in self.model.bse],
                                    'z': [round(v, 3) for v in self.model.tvalues],
                                    'pvalue': [round(v, 3) for v in self.model.pvalues]})
        self.detail['std_data'] = df_xtrain.std()
        self.detail['feature_importance'] = abs(self.detail['coef']) * self.detail['std_data']

        return self

    def predict(self, df_xtest, **kwargs):
        return self.model.predict(sm.add_constant(df_xtest))

    def predict_proba(self, df_xtest, **kwargs):
        yprob = self.model.predict(sm.add_constant(df_xtest))
        res = np.zeros((len(df_xtest), 2))
        res[:, 1] = yprob
        res[:, 0] = 1 - yprob
        return res

    def summary(self):
        print(self.detail)

    def get_importance(self):
        return self.detail.drop('const', axis=0)

#适用IV筛选入模特征

class IVSelector(TransformerMixin):
    def __init__(self):
        self.detail = None
        self.selected_features = list()

    def fit(self, df_xtrain, df_ytrain, **kwargs):
        iv_threshold = kwargs.get('iv_threshold', 0.02)
        n_jobs = kwargs.get('n_jobs', -1)
        feature_list = sorted(kwargs.get('feature_list', df_xtrain.columns.tolist()))

        # compute IV

        df_iv = kwargs.get('df_iv', None)
        if df_iv is None:
            if len(df_ytrain) > 100000:
                lst_iv = Parallel(n_jobs=n_jobs)(
                    delayed(Metrics.get_iv)(df_ytrain, df_xtrain[c]) for c in feature_list)
            else:
                lst_iv = [Metrics.get_iv(df_ytrain, df_xtrain[c]) for c in feature_list]
            df_iv = pd.DataFrame({'var': feature_list, 'iv': lst_iv})

        # select feature with IV >= iv_threshold

        df_iv.loc[:, 'selected'] = df_iv['iv'] >= iv_threshold

        self.detail = df_iv
        self.selected_features = df_iv.loc[(df_iv['selected'] == True), 'var'].tolist()

        return self

    def transform(self, df_xtest, **kwargs):
        feature_list = kwargs.get('feature_list', df_xtest.columns.tolist())
        feature_list = sorted(set(feature_list) & set(self.selected_features))
        return df_xtest[feature_list]

    def summary(self):
        print('selected features:')
        print(self.selected_features)
        print('summary')
        print(self.detail)


model_iv = IVSelector()
model_iv.fit(df_xtrain, df_ytrain)

print (model_iv.detail)

summary = pd.concat([model_vif.detail, model_iv.detail], axis=1).sort_values(by=['iv', 'vif'], ascending=[False, True])
