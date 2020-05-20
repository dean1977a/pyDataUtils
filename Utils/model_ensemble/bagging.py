#导入数据
#进行特征选择后的数据
feature_select_df = pd.read_csv('F:/Data/WEJK/feature_select_df.csv', encoding='utf-8')
#排序特征
rank_df = pd.read_csv('F:/Data/WEJK/rank_feature.csv', encoding='utf-8')
#离散特征
discrete_df = pd.read_csv('F:/Data/WEJK/discrete_feature.csv', encoding='utf-8')
print('feature_select_df.shape:{}'.format(feature_select_df.shape))
print('rank_df.shape:{}'.format(rank_df.shape))
print('discrete_df.shape:{}'.format(discrete_df.shape))

#合并数据集
df = pd.merge(feature_select_df, rank_df, on='uid', how='left')
df = pd.merge(df, discrete_df, on='uid', how='left')
df.shape



feature1 = list(feature_select_df.columns)
feature2 = set(list(rank_df.columns)+list(discrete_df.columns))
feature2.remove('uid')
feature2 = list(feature2)
random.shuffle(feature2)

x = df.drop(['label','uid'], axis=1)
y = df['label']
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=0)

#定义lightgbm的bagging函数
def bagging_lightgbm(random_seed, n_features):
    #数据集准备
    random.shuffle(feature2)
    select_feature = [x for x in feature1 if x not in ['uid','label']] + feature2[:n_features]
    x_train = train_x.loc[:, select_feature]
    x_test = test_x.loc[:, select_feature]
    #模型训练
    params = {'boosting_type': 'gbdt',
              'metric': 'auc',
              'num_leaves': 2,
              'min_child_sample':18,
              'colsample_bytree':0.5,
              'max_depth': -1,
              'learning_rate': 0.15}
    model = lgb.LGBMClassifier(**params)
    model.fit(x_train, train_y, eval_set=[(x_train, train_y), (x_test, test_y)], eval_metric='auc', early_stopping_rounds=10)
    #结果预测
    model_pre = list(model.predict_proba(x_test)[:,1])
    return model_pre

#打乱数据
random_seed = list(range(1000))
n_features = list(range(150, 300, 2))
random.shuffle(random_seed)
random.shuffle(n_features)

start = time.time()
model_pre_list = []
for i in range(10):
    model_pre = bagging_lightgbm(random_seed=random_seed[i],
                                 n_features=n_features[i])
    model_pre_list.append(model_pre)
end = time.time()
print('运行时间：{}'.format(round(end-start, 0)))


#对子模型结果求均值，得到bagging模型最终结果
bagging_prep = list(np.sum(model_pre_list, axis=0)/10)
plot_roc(test_y, bagging_prep)