#包含了训练及绘图函数
def KS(Y, Y_prob):
    fpr, tpr, threshold = roc_curve(Y, Y_prob)
    subs = abs(fpr - tpr)
    loc = np.argmax(subs)
    ks = subs[loc]
    return (fpr, tpr, ks, threshold[loc])


def plot_ROC(Y_train=None, Y_train_prob=None, Y_test=None, Y_test_prob=None, Y_oot=None, Y_oot_prob=None):
    
    if not ((Y_train is None) | (Y_train_prob is None)):
        auc_train = roc_auc_score(Y_train, Y_train_prob)
        fpr_train, tpr_train, ks_train, threshold = KS(Y_train, Y_train_prob)
        plt.plot(fpr_train, tpr_train, label='Train')
        print('Train KS:', ks_train)
        print('Auc_train: {}'.format(auc_train))
    if not ((Y_test is None) | (Y_test_prob is None)):
        auc_test = roc_auc_score(Y_test, Y_test_prob)
        fpr_test, tpr_test, ks_test, threshold = KS(Y_test, Y_test_prob)
        plt.plot(fpr_test, tpr_test, label='Test')
        print('Test KS:', ks_test)
        print('Auc_test: {}'.format(auc_test))
    if not ((Y_oot is None) | (Y_oot_prob is None)):
        auc_oot = roc_auc_score(Y_oot,Y_oot_prob)
        fpr_oot, tpr_oot, ks_oot, threshold = KS(Y_oot, Y_oot_prob)
        plt.plot(fpr_oot, tpr_oot, label='oot')
        print('oot_val KS:', ks_oot)
        print('Auc_oot_val: {}'.format(auc_oot))
    #     plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='best')
    plt.show()


def score(xbeta):
    score = 600 + 20 * (math.log2((1 - xbeta) / xbeta))  # 好人的概率/坏人的概率
    return score


def get_cut_point(values,n_bins,method):
    df = pd.DataFrame()
    df['value'] = values
    if(method == 'step'):
        df['bin'] = pd.cut(df['value'],n_bins).cat.codes
    elif(method == 'quantile'):
        df['bin'] = pd.qcut(df['value'],n_bins).cat.codes
    else:
        print("method = 'step' or 'quantile'")
    cut_point = df.groupby('bin')['value'].apply(lambda x: x.max())
    cut_point =[-9999]+list(cut_point.values)
    cut_point[len(cut_point)-1] = 9999
    return cut_point

def bin_cut(refer_values,target_values,n_bins,method):
    cut_point = get_cut_point(refer_values,n_bins,method)
    df = pd.DataFrame()
    df['value'] = target_values
    df['bin'] = pd.cut(df['value'],cut_point,labels=range(0,n_bins))
    return df['bin']



def BadRate(y, y_bin, asc=True):
    badrate_all = y.sum() / len(y)
    result = pd.DataFrame()
    bin_name = list(set(list(y_bin)))
    bad_rate = []
    percentage = []
    lift = []
    acc_bad_rate = []
    acc_lift = []

    def _acc_BadRate(y, y_bin, i, asc=True):
        if asc:
            if len(y[y_bin <= i]) == 0:
                return 0
            else:
                return len(y[(y_bin <= i) & (y == 1)]) / len(y[y_bin <= i])
        else:
            if len(y[y_bin >= i]) == 0:
                return 0
            else:
                return len(y[(y_bin >= i) & (y == 1)]) / len(y[y_bin >= i])

    for i in bin_name:
        if len(y[y_bin == i]) == 0:
            bad_rate.append(0)
            lift.append(0)
        else:
            bad_rate_i = len(y[(y_bin == i) & (y == 1)]) / len(y[y_bin == i])
            bad_rate.append(bad_rate_i)
            lift.append(bad_rate_i / badrate_all)

        percentage.append(len(y[y_bin == i]) / len(y))
        acc_bad_rate.append(_acc_BadRate(y, y_bin, i, asc))
        acc_lift.append(_acc_BadRate(y, y_bin, i, asc) / badrate_all)

    result['bin'] = bin_name
    result['bad_rate'] = bad_rate
    result['acc_bad_rate'] = acc_bad_rate
    result['percentage'] = percentage
    result['lift'] = lift
    result['acc_lift'] = acc_lift

    result = result.sort_values(by='bin').reset_index(drop=True)
    return result


def PlotKS(preds, labels, n, asc):
    # preds is score: asc=1
    # preds is prob: asc=0

    pred = preds  # 预测值
    bad = labels  # 取1为bad, 0为good
    ksds = pd.DataFrame({'bad': bad, 'pred': pred})
    ksds['good'] = 1 - ksds.bad

    if asc == 1:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, True])
    elif asc == 0:
        ksds1 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, True])
    ksds1.index = range(len(ksds1.pred))
    ksds1['cumsum_good1'] = 1.0 * ksds1.good.cumsum() / sum(ksds1.good)
    ksds1['cumsum_bad1'] = 1.0 * ksds1.bad.cumsum() / sum(ksds1.bad)

    if asc == 1:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[True, False])
    elif asc == 0:
        ksds2 = ksds.sort_values(by=['pred', 'bad'], ascending=[False, False])
    ksds2.index = range(len(ksds2.pred))
    ksds2['cumsum_good2'] = 1.0 * ksds2.good.cumsum() / sum(ksds2.good)
    ksds2['cumsum_bad2'] = 1.0 * ksds2.bad.cumsum() / sum(ksds2.bad)

    # ksds1 ksds2 -> average
    ksds = ksds1[['cumsum_good1', 'cumsum_bad1']]
    ksds['cumsum_good2'] = ksds2['cumsum_good2']
    ksds['cumsum_bad2'] = ksds2['cumsum_bad2']
    ksds['cumsum_good'] = (ksds['cumsum_good1'] + ksds['cumsum_good2']) / 2
    ksds['cumsum_bad'] = (ksds['cumsum_bad1'] + ksds['cumsum_bad2']) / 2

    # ks
    ksds['ks'] = ksds['cumsum_bad'] - ksds['cumsum_good']
    ksds['tile0'] = range(1, len(ksds.ks) + 1)
    ksds['tile'] = 1.0 * ksds['tile0'] / len(ksds['tile0'])

    qe = list(np.arange(0, 1, 1.0 / n))
    qe.append(1)
    qe = qe[1:]

    ks_index = pd.Series(ksds.index)
    ks_index = ks_index.quantile(q=qe)
    ks_index = np.ceil(ks_index).astype(int)
    ks_index = list(ks_index)

    ksds = ksds.loc[ks_index]
    ksds = ksds[['tile', 'cumsum_good', 'cumsum_bad', 'ks']]
    ksds0 = np.array([[0, 0, 0, 0]])
    ksds = np.concatenate([ksds0, ksds], axis=0)
    ksds = pd.DataFrame(ksds, columns=['tile', 'cumsum_good', 'cumsum_bad', 'ks'])

    ks_value = ksds.ks.max()
    ks_pop = ksds.tile[ksds.ks.idxmax()]
    print('ks_value is ' + str(np.round(ks_value, 4)) + ' at pop = ' + str(np.round(ks_pop, 4)))

    # chart
    plt.plot(ksds.tile, ksds.cumsum_good, label='cum_good',
             color='blue', linestyle='-', linewidth=2)

    plt.plot(ksds.tile, ksds.cumsum_bad, label='cum_bad',
             color='red', linestyle='-', linewidth=2)

    plt.plot(ksds.tile, ksds.ks, label='ks',
             color='green', linestyle='-', linewidth=2)

    plt.axvline(ks_pop, color='gray', linestyle='--')
    plt.axhline(ks_value, color='green', linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_good'], color='blue', linestyle='--')
    plt.axhline(ksds.loc[ksds.ks.idxmax(), 'cumsum_bad'], color='red', linestyle='--')
    plt.title('KS=%s ' % np.round(ks_value, 4) +
              'at Pop=%s' % np.round(ks_pop, 4), fontsize=15)

    return ksds

def PlotBadrate(df):
    '''
    df: BadRate()的返回值 或相同列名、格式的pd.DataFrame()
    '''
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['SimHei']
    fig = plt.figure(figsize=(20,10))
    ax1 = fig.add_subplot(111)
    ax1.set_ylim(0,0.1)
    ax1.set_yticks(np.arange(0,0.07,0.01))
    ax1.set_ylabel('BadRate',fontsize=18)
    ax1.set_xlabel('score')
    plt.plot(df['bin'],df['bad_rate'])

    for a,b in zip(df['bin'],df['bad_rate']):  
        plt.text(a,b+0.03,'%.3f' % b ,color='black',fontsize=10,ha='center',va='bottom')  #将数值显示在图形上

    ax2 = ax1.twinx()
    ax2.set_ylim(0,0.5)
    ax2.set_yticks(np.arange(0,0.3,0.05))
    ax2.set_ylabel('percentage',fontsize=18)

    plt.bar(df['bin'],df['percentage'],alpha=0.3,color='blue',label='percentage')

    for a,b in zip(df['bin'],df['percentage']):  
        plt.text(a,b+0.005,'%.3f' % b ,color='blue',fontsize=10,ha='center',va='bottom')  #将数值显示在图形上

    fig.show()
    
def lgb_train(x_train, y_train, x_test, y_test,x_oot,y_oot ,params = {},is_same_model = False ,model_save_path=""):
    '''
    模型的训练和保存
    :param x_train:
    :param y_train:
    :param q_train:
    :param model_save_path:
    :return:
    '''
    lgb_train=lgb.Dataset(x_train,y_train)
    lgb_test=lgb.Dataset(x_test,y_test)
    lgb_oot=lgb.Dataset(x_oot,y_oot)

    model_lgb=lgb.train(train_set=lgb_train,
                       early_stopping_rounds=500,
                       num_boost_round=1000,
                       params= params,
                       valid_sets=lgb_test,
                        verbose_eval=10)
    importance = model_lgb.feature_importance(importance_type='split')
    feature_name = model_lgb.feature_name()
    feature_importance=pd.DataFrame({'name':feature_name,
                                    'importance':importance}).sort_values(by=['importance'],ascending=False)
    total = feature_importance['importance'].sum()
    feature_importance['imp_ratio'] = feature_importance['importance'].apply(lambda x : x/total)
    feature_importance = feature_importance.reset_index(drop=True)
    
    Y_train_prob=model_lgb.predict(X_train)
    Y_test_prob=model_lgb.predict(X_test)
    Y_oot_prob=model_lgb.predict(X_oot)
    print("变量个数：",X_train.shape[1])
    
    plot_ROC(Y_train=Y_train,Y_train_prob=Y_train_prob,Y_test=Y_test,Y_test_prob=Y_test_prob,Y_oot=Y_oot,Y_oot_prob=Y_oot_prob)

    df_train = pd.DataFrame()
    df_test = pd.DataFrame()
    df_oot = pd.DataFrame()

    df_train['prob'] = Y_train_prob
    df_test['prob'] = Y_test_prob
    df_oot['prob'] = Y_oot_prob

    df_train['score'] = df_train['prob'].apply(lambda x : score(x))
    df_test['score']= df_test['prob'].apply(lambda x: score(x))
    df_oot['score'] = df_oot['prob'].apply(lambda x: score(x))

    df_train['Y'] = Y_train.values
    df_test['Y'] = Y_test.values
    df_oot['Y'] = Y_oot.values

    df_train['bin'] = bin_cut(df_train['score'],df_train['score'],10,'quantile')
    df_test['bin'] = bin_cut(df_train['score'],df_test['score'],10,'quantile')
    df_oot['bin'] = bin_cut(df_test['score'],df_oot['score'],10,'quantile')

    eva_df_train = BadRate(df_train['Y'],df_train['bin'])
    eva_df_oot = BadRate(df_oot['Y'],df_oot['bin'])
    eva_df_test = BadRate(df_test['Y'],df_test['bin'])
    print("train: \n")
    print(eva_df_train)
    print("test: \n")
    print(eva_df_test)
    print("oot: \n")
    print(eva_df_oot)

    if is_same_model == True:
        model_lgb.save_model(model_save_path)
        
    return model_lgb,feature_importance
