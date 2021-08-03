#用于创建sklearn API接口的多个二分类模型stacking结果
def get_sklearn_classfiy_stacking(clf, train_feature, test_feature, score, model_name, class_number, n_folds, train_num, test_num):
    print('\n****开始跑', model_name, '****')
    stack_train = np.zeros((train_num, class_number))
    stack_test = np.zeros((test_num, class_number))
    score_mean = []
    skf = StratifiedKFold(n_splits=n_folds, random_state=1017)
    tqdm.desc = model_name
    for i, (tr, va) in enumerate(skf.split(train_feature, score)):
        clf.fit(train_feature[tr], score[tr])
        score_va = clf._predict_proba_lr(train_feature[va])
        score_te = clf._predict_proba_lr(test_feature)
        score_single = roc_auc_score(score[va], clf._predict_proba_lr(train_feature[va])[:, 1])
        score_mean.append(np.around(score_single, 5))
        stack_train[va] += score_va
        stack_test += score_te
    stack_test /= n_folds
    stack = np.vstack([stack_train, stack_test])
    df_stack = pd.DataFrame()
    for i in range(stack.shape[1]):
        df_stack['tfidf_' + model_name + '_classfiy_{}'.format(i)] = stack[:, i]
    print(model_name, '处理完毕')
    return df_stack, score_mean

#示例
model_list = [
    ['LogisticRegression', LogisticRegression(random_state=1017, C=3)],
    ['SGDClassifier', SGDClassifier(random_state=1017, loss='log')],
    ['PassiveAggressiveClassifier', PassiveAggressiveClassifier(random_state=1017, C=2)],
    ['RidgeClassfiy', RidgeClassifier(random_state=1017)],
    ['LinearSVC', LinearSVC(random_state=1017)]
]

feature = pd.DataFrame()
for i in model_list:
    stack_result, score_mean = get_sklearn_classfiy_stacking(i[1], train_feature, test_feature, score, i[0], 2, 5, len(df_train), len(df_test))
    feature = pd.concat([feature, stack_result], axis=1, sort=False)
    print('五折结果', score_mean)
    print('平均结果', np.mean(score_mean))
    result = stack_result[len(df_train):]
    put_result = pd.DataFrame()
    put_result['ID'] = df_test['ID']
    put_result['Pred'] = list(result['tfidf_' + i[0] + '_classfiy_{}'.format(1)])
    put_result.to_csv('result/result_' + i[0] + '_' + str(np.around(np.mean(score_mean), 5)) + '.csv', index=False)
feature.to_csv('features/tfidf_classfiy_stacking.csv', index=False)
