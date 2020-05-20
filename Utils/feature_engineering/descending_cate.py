#https://zhuanlan.zhihu.com/p/122857572
# 分类型变量的降基处理
def descending_cate(df,col_list,threshold=None):
    """
    df: 数据集
    col_list:变量list集合
    threshold:降基处理的阈值

    return :处理后的数据集
    """
    df2 = df.copy()
    for col in col_list:
        value_series = df[col].value_counts()/df[df[col].notnull()].shape[0]
        small_value = []
        for value_name,value_pct in zip(value_series.index,value_series.values):
            if value_pct<=threshold:
                small_value.append(value_name)
        df2.loc[df2[col].isin(small_value),col]='other'
    return df2