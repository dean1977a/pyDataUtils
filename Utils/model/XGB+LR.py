import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics
import logging
import xgboost as xgb
import time
from sklearn.datasets import load_iris

logging.basicConfig(format='%(asctime)s : %(levelname)s: %(message)s', level=logging.INFO)


def XGBoost_LR(df_train):
    X_train = df_train.values[:, :-1]
    y_train = df_train.values[:, -1]
    X_train_xgb, X_train_lr, y_train_xgb, y_train_lr = train_test_split(X_train, 
                                                y_train, test_size=0.75)
    XGB = xgb.XGBClassifier(n_estimators = 6）

    XGB.fit(X_train_xgb, y_train_xgb)
    logging.info("训练集特征数据为： \n%s" % X_train_xgb)
    logging.info("训练集标签数据为： \n%s" % y_train_xgb)
    logging.info("转化为叶子节点后的特征为：\n%s" % XGB.apply(X_train_xgb, ntree_limit=0))

    XGB_enc = OneHotEncoder()
    XGB_enc.fit(XGB.apply(X_train_xgb, ntree_limit=0)) # ntree_limit 预测时使用的树的数量
    XGB_LR = LogisticRegression()
    XGB_LR.fit(XGB_enc.transform(XGB.apply(X_train_lr)), y_train_lr.astype('int'))
    X_predict = XGB_LR.predict_proba(XGB_enc.transform(XGB.apply(X_train)))[:, 1]
    AUC_train = metrics.roc_auc_score(y_train.astype('int'), X_predict)
    logging.info("AUC of train data is %f" % AUC_train)


if __name__ == "__main__":
    start = time.clock()
    #加载数据集
    iris=load_iris()
    df_data = pd.concat([pd.DataFrame(iris.data), pd.DataFrame(iris.target)], axis=1)
    df_data.columns = ["x1","x2", "x3","x4", "y"]
    df_new = df_data[df_data["y"]<2]
    logging.info("Train model begine...")
    XGBoost_LR(df_new)
    end = time.clock()
    logging.info("Program over, time cost is %s" % (end-start))



#!/usr/bin/env python
#encoding=utf8
'''
  Author: zldeng
  create@2017-09-21 11:23:12
'''
 
import sys,os
reload(sys)
sys.setdefaultencoding('utf8')
 
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression as LR

import xgboost as xgb
import numpy as np
import pickle

from xgboost import XGBClassifier, DMatrix

class XGBoostLR(object):
    '''
    xboost as feature transform.
    xgboost'output is the input feature of LR model
    '''
    def __init__(self,xgb_model_name,lr_model_name,
            one_hot_encoder_model_name,
            xgb_eval_metric = 'mlogloss',xgb_nthread = 32,n_estimators = 100):
        self.xgb_model_name = xgb_model_name
        self.lr_model_name = lr_model_name
        self.one_hot_encoder_model_name = one_hot_encoder_model_name
        self.xgb_eval_metric = xgb_eval_metric
        self.xgb_nthread = xgb_nthread

        self.init_flag = False

    def trainModel(self,train_x,train_y):
        #train a xgboost model
        sys.stdout.flush()
        self.xgb_clf = xgb.XGBClassifier(nthread = self.xgb_nthread)
        self.xgb_clf.fit(train_x,train_y,eval_metric = self.xgb_eval_metric,
                eval_set = [(train_x,train_y)])

        xgb_eval_result = self.xgb_clf.evals_result()
        logging.info("GB_train eval_result： \n%s" % xgb_eval_result)
        
        sys.stdout.flush()

        train_x_mat = DMatrix(train_x)
        logging.info("get boost tree leaf info... ")
        train_xgb_pred_mat = self.xgb_clf.get_booster().predict(train_x_mat,
                pred_leaf = True)
        logging.info("get boost tree leaf info done ")
        
        logging.info("begin one-hot encoding... ")

        self.one_hot_encoder = OneHotEncoder()
        train_lr_feature_mat = self.one_hot_encoder.fit_transform(train_xgb_pred_mat)

        logging.info("one-hot encoding done! ")
        logging.info("train_mat \n%s" % train_lr_feature_mat.shape)
        
        sys.stdout.flush()
        #train a LR model
        self.lr_clf = LR()
        self.lr_clf.fit(train_lr_feature_mat,train_y)
        
        self.init_flag = True
        
        logging.info("dump xgboost+lr model.. ")
        
        pickle.dump(self.xgb_clf,file(self.xgb_model_name,'wb'),True)
        pickle.dump(self.lr_clf,file(self.lr_model_name,'wb'),True)
        pickle.dump(self.one_hot_encoder,file(self.one_hot_encoder_model_name,'wb'),True)
        
        logging.info("Train xgboost and lr model done")
        
    
    def loadModel(self):
        try:
            self.xgb_clf = pickle.load(file(self.xgb_model_name,'rb'))
            self.lr_clf = pickle.load(file(self.lr_model_name,'rb'))
            self.one_hot_encoder = pickle.load(file(self.one_hot_encoder_model_name,'rb'))

            self.init_flag = True
        except Exception,e:
        	logging.info("Load XGB and LR model fail. +"  + str(e))
            sys.exit(1)
    
    def testModel(self,test_x,test_y):
        if not self.init_flag:
            self.loadModel()
        
        test_x_mat = DMatrix(test_x)
        xgb_pred_mat = self.xgb_clf.get_booster().predict(test_x_mat,pred_leaf = True)
        
        lr_feature = self.one_hot_encoder.transform(xgb_pred_mat)
        #print 'test_mat:',lr_feature.shape

        lr_pred_res = self.lr_clf.predict(lr_feature)

        total = len(test_y)
        correct = 0

        for idx in range(total):
            if lr_pred_res[idx] == test_y[idx]:
                correct += 1

        logging.info("XGB+LR test: \n%s" total,correct,correct*1.0/total)
        