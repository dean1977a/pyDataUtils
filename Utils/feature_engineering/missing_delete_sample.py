#https://zhuanlan.zhihu.com/p/122857572
# 缺失值剔除（单个样本）
def missing_delete_sample(df,threshold=None):
    """
    df:数据集
    threshold:缺失个数删除的阈值

    return :删除缺失后的数据集
    """
    df2 = df.copy()
    missing_series = df.isnull().sum(axis=1)
    missing_list = list(missing_series)
    missing_index_list = []
    for i,j in enumerate(missing_list):
        if j>=threshold:
            missing_index_list.append(i)
    df2 = df2[~(df2.index.isin(missing_index_list))]
    print('缺失变量个数在{}以上的样本数有{}个'.format(threshold,len(missing_index_list)))
    return df2