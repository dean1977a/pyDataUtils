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

def plot_ks(y_true, y_pred_proba, output_path=None):
    """Plot K-S curve of a model
    Parameters
    ----------
    y_true: numpy.array, shape (number of examples,)
            The target column (or dependent variable).  
    
    y_pred_proba: numpy.array, shape (number of examples,)
            The score or probability output by the model. The probability
            of y_true being 1 should increase as this value
            increases.
            If Scorecard model's parameter "PDO" is negative, then the higher the 
            model scores, the higher the probability of y_pred being 1. This Function
            works fine. 
            However!!! if the parameter "PDO" is positive, then the higher 
            the model scores, the lower the probability of y_pred being 1. In this case,
            just put a negative sign before the scores array and pass `-scores` as parameter
            y_pred_proba of this function. 
    
    output_path: string, optional(default=None)
        the location to save the plot. 
        e.g. r'D:\\Work\\jupyter\\'.
    """    
    # Check input data 
    if isinstance(y_true, pd.Series):
        target = y_true.values
    elif isinstance(y_true, np.ndarray):
        target = y_true
    else:
        raise TypeError('y_true should be either numpy.array or pandas.Series')

    if isinstance(y_pred_proba, pd.Series):
        scores = y_pred_proba.values
    elif isinstance(y_pred_proba, np.ndarray):
        scores = y_pred_proba
    else:
        raise TypeError('y_pred_proba should be either numpy.array or pandas.Series')

    # Group scores into 10 groups ascendingly
    interval_index = pd.IntervalIndex(pd.qcut(
        pd.Series(scores).sort_values(ascending=False), 10, duplicates='drop'
                                              ).drop_duplicates()) 
    group = pd.Series([interval_index.get_loc(element) for element in scores])

    distribution = pd.DataFrame({'group':group,
                                 'y_true':target
                                 })
    grouped = distribution.groupby('group')
    pct_of_target = grouped['y_true'].sum() / np.sum(target)
    pct_of_nontarget = (grouped['y_true'].size() - grouped['y_true'].sum()) / (len(target) - np.sum(target))
    cumpct_of_target = pd.Series([0] + list(pct_of_target.cumsum()))
    cumpct_of_nontarget = pd.Series([0] + list(pct_of_nontarget.cumsum()))
    diff = cumpct_of_target - cumpct_of_nontarget
    
    # Plot ks curve
    plt.plot(cumpct_of_target, label='Y=1')
    plt.plot(cumpct_of_nontarget, label='Y=0')
    plt.plot(diff, label='K-S curve')
    ks = round(diff.abs().max(),3)
    print('KS = '+str(ks))
    plt.annotate(s='KS = '+str(ks) ,xy=(diff.abs().idxmax(),diff.abs().max()))
    plt.xlim((0,10))
    plt.ylim((0,1))
    plt.title('K-S Curve', fontdict=font_title)   
    plt.xlabel('Group of scores', fontdict=font_text)
    plt.ylabel('Cumulated class proportion', 
                fontdict=font_text)    
    plt.legend()

    if output_path is not None:
        plt.savefig(output_path+r'K-S_Curve.png', dpi=500, bbox_inches='tight')    
    plt.show()
