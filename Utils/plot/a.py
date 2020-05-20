#https://zhuanlan.zhihu.com/p/123352224
def plot_cate_var(df,col_list,hspace=0.4,wspace=0.4,plt_size=None,plt_num=None,x=None,y=None):
        '''
    显示类别型变量的分布情况
    '''
    plt.figure(figsize=plt_size)
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i,col in zip(range(1,plt_num+1,1),col_list):
        plt.subplot(x,y,i)
        plt.title(col)
        sns.countplot(data=df,y=col)
        plt.ylabel('')
    return plt.show()

def plot_num_col(df,col_list,hspace=0.4,wspace=0.4,plt_type=None,plt_size=None,plt_num=None,x=None,y=None):
    '''
    显示数值型变量的分布情况
    '''

    plt.figure(figsize=plt_size)
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    if plt_type=='hist':
        for i,col in zip(range(1,plt_num+1,1),col_list):
            plt.subplot(x,y,i)
            plt.title(col)
            sns.distplot(df[col].dropna())
            plt.xlabel('')
    return plt.show()


def plot_default_cate(df,col_list,target,hspace=0.4,wspace=0.4,plt_size=None,plt_num=None,x=None,y=None):
    '''
    类别型变量的违约率分析
    '''
    all_bad = df[target].sum()
    total = df[target].count()
    all_default_rate = all_bad*1.0/total
    
    plt.figure(figsize=plt_size)
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    plt.rcParams['font.sans-serif']=['Microsoft YaHei']
    plt.rcParams['axes.unicode_minus'] = False
    for i,col in zip(range(1,plt_num+1,1),col_list):
        d1 = df.groupby(col)
        d2 = pd.DataFrame()
        d2['total'] = d1[target].count()
        d2['bad'] = d1[target].sum()
        d2['default_rate'] = d2['bad']/d2['total']
        d2 = d2.reset_index()
        plt.subplot(x,y,i)
        plt.title(col)
        plt.axvline(x=all_default_rate)
        sns.barplot(data=d2,y=col,x='default_rate')
        plt.ylabel('')
    return plt.show()


def plot_default_num(df,col_list,target,hspace=0.4,wspace=0.4,q=None,plt_size=None,plt_num=None,x=None,y=None):
    '''
    数值型变量的违约率分析
    '''
    all_bad = df[target].sum()
    total = df[target].count()
    all_default_rate = all_bad*1.0/total 
    
    plt.figure(figsize=plt_size)
    plt.subplots_adjust(hspace=hspace,wspace=wspace)
    for i,col in zip(range(1,plt_num+1,1),col_list):
        bucket = pd.qcut(df[col],q=q,duplicates='drop')
        d1 = df.groupby(bucket)
        d2 = pd.DataFrame()
        d2['total'] = d1[target].count()
        d2['bad'] = d1[target].sum()
        d2['default_rate'] = d2['bad']/d2['total']
        d2 = d2.reset_index()
        plt.subplot(x,y,i)
        plt.title(col)
        plt.axhline(y=all_default_rate)
        sns.pointplot(data=d2,x=col,y='default_rate',color='hotpink')
        plt.xticks(rotation=60)
        plt.xlabel('')
    return plt.show()