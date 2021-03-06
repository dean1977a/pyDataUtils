data_temp = data
# 分箱：等距等频分箱
# 等距分箱
#bins=10 分箱数
data_temp['deposit_cur_balance_bins'] = binning(df = data_temp,col = 'deposit_cur_balance',method='frequency',bins=10)
data_temp['deposit_year_total_balance_bins'] = binning(df = data_temp,col = 'deposit_year_total_balance',method='frequency',bins=10)


def binning(df,col,method,bins):
    # 需要判断类型
    uniqs=df[col].nunique()
    if uniqs<=bins:
        raise KeyError('nunique is smaller than bins: '+col)
    #print('nunique is smaller than bins: '+col)
        return
    # 左开右闭
    def ff(x,fre_list):
        if x<=fre_list[0]:
            return 0
        elif x>fre_list[-1]:
            return len(fre_list)-1
        else :
            for i in range(len(fre_list)-1):
                if x>fre_list[i] and x<=fre_list[i+1]:
                    return i
    # 等距分箱
    if method=='distance':
        umax=np.percentile(df[col],99.99)
        umin=np.percentile(df[col],0.01)
        step=(umax-umin)/bins
        fre_list=[umin+i*step for i in range(bins+1)]
        return df[col].map(lambda x:ff(x,fre_list))
    # 等频分箱
    elif method=='frequency' :
        fre_list=[np.percentile(df[col],100/bins*i) for i in range(bins+1)]
        fre_list=sorted(list(set(fre_list)))
        return df[col].map(lambda x:ff(x,fre_list))


