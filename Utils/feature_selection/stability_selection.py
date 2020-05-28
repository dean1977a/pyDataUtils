

def feature_stability_selection(model,df,)
    '''
	特征稳定性
	https://thuijskens.github.io/2018/07/25/stability-selection/
	https://zhuanlan.zhihu.com/p/110643632
	'''

	from sklearn.linear_model import LogisticRegression
	from sklearn.pipeline import Pipeline
	from sklearn.preprocessing import StandardScaler
	from stability_selection import StabilitySelection

    if model == 'LogisticRegression':
    	clf = LogisticRegression(penalty='l1', class_weight='balanced', solver='auto',random_state= )

    if model == 'LogisticRegression':
    	clf = RidgeClassifier(alpha=1.0, class_weight='balanced', solver='auto', random_state=None)

	self.model = clf

	base_estimator = Pipeline([
        ('scaler', StandardScaler()),
        ('model', model)
    ])

    selector = StabilitySelection(base_estimator=base_estimator, lambda_name='model__C',
                                  lambda_grid=np.logspace(-5, -1, 50))
    selector.fit(X, y)

    fig, ax = plot_stability_path(selector)
    fig.show()

    selected_variables = selector.get_support(indices=True)
    selected_scores = selector.stability_scores_.max(axis=1)

    print('Selected variables are:')
    print('-----------------------')

    for idx, (variable, score) in enumerate(zip(selected_variables, selected_scores[selected_variables])):
        print('Variable %d: [%d], score %.3f' % (idx + 1, variable, score))



def stepwise_feature_selection(model,df,original_columns,target):

    train_columns = list(train.columns[13:])
    usefull_columns = []
    not_usefull_columns = []
    best_score = 0
    
    train_tmp = train[original_columns]
    print('Training with {} features'.format(train_tmp.shape[1]))
    x_train, x_val, y_train, y_val = train_test_split(train_tmp, target, test_size = 0.2, random_state = 42)
    xg_train = lgb.Dataset(x_train, label = y_train)
    xg_valid = lgb.Dataset(x_val, label= y_val)
    clf = lgb.train(param, xg_train, 100000, valid_sets = [xg_train, xg_valid], verbose_eval = 3000, 
                    early_stopping_rounds = 100)
    predictions = clf.predict(x_val)
    rmse_score = np.sqrt(mean_squared_error(y_val, predictions))
    print("RMSE baseline val score: ", rmse_score)
    best_score = rmse_score
    
    for num, i in enumerate(train_columns):
        train_tmp = train[original_columns + usefull_columns + [i]]
        print('Training with {} features'.format(train_tmp.shape[1]))
        x_train, x_val, y_train, y_val = train_test_split(train_tmp, target, test_size = 0.2, random_state = 42)
        
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train = scaler.transform(x_train)
        x_val = scaler.transform(x_val)
        print( x_train.shape, x_val.shape, y_train.shape )

        if model == 'LogisticRegression':
            clf = LogisticRegression(penalty='l1')
       
        if model == 'RidgeClassifier':
            clf = Ridge(alpha=20, copy_X=True, fit_intercept=True, solver='auto',max_iter=10000,normalize=False, random_state=0,  tol=0.0025)

        clf.fit(x_train,y_train)
        
        predictions = clf.predict(x_val)
        
        auc_score = roc_auc_score(val_y, oof_preds[val_idx]
        

        
        if auc_score  >= best_score + 0.01:
        	print('Column {} is usefull'.format(i))
            best_score = auc_score
            usefull_columns.append(i)
        else:
            print('Column {} is not usefull'.format(i))
            not_usefull_columns.append(i)
            
        print('Best  score for iteration {} is {}'.format(num + 1, best_score))

        return usefull_columns, not_usefull_columns
            
usefull_columns, not_usefull_columns = auto_feature_selection(model,df,original_columns,target)