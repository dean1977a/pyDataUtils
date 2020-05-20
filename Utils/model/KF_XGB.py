# Define your kfold cross validation
# We used 3 folds, because we did not see improvements with higher folds
# We are not scared of shuffling because The whole point of this comp is to be independent of time. Test is shuffled
# 未完成的
# trainModel 需要重写
class kf_xgb(object):
        '''
    xboost as feature transform.
    xgboost'output is the input feature of LR model
    '''
    def __init__(self,xgb_model_name,lr_model_name,
            one_hot_encoder_model_name,
            xgb_eval_metric = 'mlogloss',xgb_nthread = 16,n_estimators = 100,n_fold = 3):
        self.xgb_model_name = xgb_model_name
        self.lr_model_name = lr_model_name
        self.one_hot_encoder_model_name = one_hot_encoder_model_name
        self.xgb_eval_metric = xgb_eval_metric
        self.xgb_nthread = xgb_nthread
        self.n_fold = n_fold


    #这里开始都要重写
    X = train.drop(['id', 'target'], axis=1).values
    y = train.target.values
    test_id = test.id.values
    test = test.drop('id', axis=1)
    sub = pd.DataFrame()
    sub['id'] = test_id
    sub['target'] = np.zeros_like(test_id)

    def trainModel(self,train_x,train_y):
        kf = KFold(n_splits=self.n_fold, shuffle=True, random_state=1337)
        kf = list(kf.split(np.arange(len(train_sample))))

        oof_LGBM = np.zeros(len(train_sample))
        sub_LGBM = np.zeros(len(submission))
        seeds = [0,1,2,3,4,5,6,7,8,9]

        for seed in seeds:
            print('Seed',seed)
            for fold_n, (train_index, valid_index) in enumerate(kf):
                print('Fold', fold_n)

                # Create train and validation data using only LGBM_feats.
                # 训练集和验证集这里需要修改

                X_train, X_valid = X[train_index], X[test_index]
                y_train, y_valid = y[train_index], y[test_index]
                # Convert our data into XGBoost format
                d_train = xgb.DMatrix(X_train, y_train)
                d_valid = xgb.DMatrix(X_valid, y_valid)
                d_test = xgb.DMatrix(test.values)
                watchlist = [(d_train, 'train'), (d_valid, 'valid')]

                # More parameters has to be tuned. Good luck :)
                params = {
                    'min_child_weight': 10.0,
                    'objective': 'binary:logistic',
                    'max_depth': 7,
                    'max_delta_step': 1.8,
                    'colsample_bytree': 0.4,
                    'subsample': 0.8,
                    'eta': 0.025,
                    'gamma': 0.65,
                    'num_boost_round': 700
                }

                clf = xgb.train(params, d_train, 1600, watchlist, early_stopping_rounds=70, feval=gini_xgb, maximize=True, verbose_eval=100)
                print('[Fold %d/%d Prediciton:]' % (i + 1, kfold))

                oof_LGBM[valid_index] += clf.predict(train_sample[LGBM_feats].iloc[valid_index], num_iteration=clf.best_iteration)
                sub_LGBM += clf.predict(test[LGBM_feats], num_iteration=clf.best_iteration) / n_fold

                del X_train, X_valid,y_train, y_valid,d_train,d_valid,d_test
                gc.collect()
                
        oof_LGBM = oof_LGBM / len(seeds)
        sub_LGBM = sub_LGBM / len(seeds)
            
        print('\nMAE for LGBM: ', mean_absolute_error(targets['target'], oof_LGBM))
