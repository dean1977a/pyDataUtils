#https://zhuanlan.zhihu.com/p/122857572
def fillna_num_var(df,col_list,fill_type=None,filled_df=None):
    """
    缺失率 5%以下：中位数
    缺失率 5%-15%:随机森林填充
    缺失率 15%以上：当作一个类别
    df:数据集
    col_list:变量list集合
    fill_type:填充方式：中位数/随机森林/当做一个类别
    filled_df :已填充好的数据集，当填充方式为随机森林时 使用

    return:已填充好的数据集
    """
    df2 = df.copy()
    for col in col_list:
        if fill_type=='median':
            df2[col] = df2[col].fillna(df2[col].median())
        if fill_type=='class':
            df2[col] = df2[col].fillna(-999)
        if fill_type=='rf':
            rf_df = pd.concat([df2[col],filled_df],axis=1)
            known = rf_df[rf_df[col].notnull()]
            unknown = rf_df[rf_df[col].isnull()]
            x_train = known.drop([col],axis=1)
            y_train = known[col]
            x_pre = unknown.drop([col],axis=1)
            rf = RandomForestRegressor(random_state=0)
            rf.fit(x_train,y_train)
            y_pre = rf.predict(x_pre)
            df2.loc[df2[col].isnull(),col] = y_pre
    return df2



