# Define your kfold cross validation
# We used 3 folds, because we did not see improvements with higher folds
# We are not scared of shuffling because The whole point of this comp is to be independent of time. Test is shuffled

def KF_LGB():
    n_fold = 3
    kf = KFold(n_splits=n_fold, shuffle=True, random_state=1337)
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
            trn_data = lgb.Dataset(train_sample[LGBM_feats].iloc[train_index], label=targets['target'].iloc[train_index])
            val_data = lgb.Dataset(train_sample[LGBM_feats].iloc[valid_index], label=targets['target'].iloc[valid_index])

            params = {'num_leaves': 4, # Low number of leaves reduces LGBM complexity
              'min_data_in_leaf': 5,
              'objective':'binary', # Fitting to fair objective performed better than fitting to MAE objective
              'max_depth': -1,
              'learning_rate': 0.01,
              "boosting": "gbdt", 
              'boost_from_average': True,
              "feature_fraction": 0.9,
              "bagging_freq": 1,
              "bagging_fraction": 0.5,
              "bagging_seed": 0,
              "metric": 'auc',
              "verbosity": -1,
              'max_bin': 500,
              'reg_alpha': 0, 
              'reg_lambda': 0,
              'seed': seed,
              'n_jobs': 1
              }

            clf = lgb.train(params, trn_data, 1000000, valid_sets = [trn_data, val_data], verbose_eval=1000, early_stopping_rounds = 1000)

            oof_LGBM[valid_index] += clf.predict(train_sample[LGBM_feats].iloc[valid_index], num_iteration=clf.best_iteration)
            sub_LGBM += clf.predict(test[LGBM_feats], num_iteration=clf.best_iteration) / n_fold
            
    oof_LGBM = oof_LGBM / len(seeds)
    sub_LGBM = sub_LGBM / len(seeds)
        
    print('\nMAE for LGBM: ', mean_absolute_error(targets['target'], oof_LGBM))