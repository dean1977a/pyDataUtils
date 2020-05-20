#https://zhuanlan.zhihu.com/p/122857572
def fillna_cate_var(df,col_list,fill_type=None):
    """
    用众数进行填充
    单独当作一个类别
    df:数据集
    col_list:变量list集合
    fill_type: 填充方式：众数/当做一个类别

    return :填充后的数据集
    """
    df2 = df.copy()
    for col in col_list:
        if fill_type=='class':
            df2[col] = df2[col].fillna('unknown')
        if fill_type=='mode':
            df2[col] = df2[col].fillna(df2[col].mode()[0])
    return df2