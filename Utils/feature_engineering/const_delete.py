#https://zhuanlan.zhihu.com/p/122857572
# 常变量/同值化处理
def const_delete(df,col_list,threshold=None):
    """
    df:数据集
    col_list:变量list集合
    threshold:同值化处理的阈值

    return :处理后的数据集
    """
    df2 = df.copy()
    const_col = []
    for col in col_list:
        const_pct = df2[col].value_counts().iloc[0]/df2[df2[col].notnull()].shape[0]
        if const_pct>=threshold:
            const_col.append(col)
    df2 = df2.drop(const_col,axis=1)
    print('常变量/同值化处理的变量个数为{}'.format(len(const_col)))
    return df2