
#https://zhuanlan.zhihu.com/p/122857572
# 单个样本的缺失分布
def plot_missing_user(df,plt_size=None):
    """
    df: 数据集
    plt_size: 图纸的尺寸

    return :缺失分布图（折线图形式）
    """
    missing_series = df.isnull().sum(axis=1)
    list_missing_num  = sorted(list(missing_series.values))
    plt.figure(figsize=plt_size)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    plt.plot(range(df.shape[0]),list_missing_num)
    plt.ylabel('缺失变量个数')
    plt.xlabel('samples')
    return plt.show()
plot_missing_user(trainData,plt_size=None)

