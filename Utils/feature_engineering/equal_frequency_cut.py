
未完成待修改。。。
def equal_frequency_cut(df,col,target,bins)
    '''
    对数据进行等频切分，并显示相关统计数据
    df:dataframe
    col:col need to  equal frequency cut
    target:label col
    bins:
    '''
	df_train.loc[:,col+'_qcut'] = pd.qcut(df_train[col], 10)
	df_train.head()
	df_train = df_train.sort_values(col)
	alist = list(set(df_train[col+'_qcut']))
	badrate = {}
	for x in alist:
	    
	    a = df_train[df_train.fare_qcut == x]
	    
	    bad = a[a.label == 1][target].count()
	    good = a[a.label == 0][target].count()
	    
	    badrate[x] = bad/(bad+good)
	f = zip(badrate.keys(),badrate.values())
	f = sorted(f,key = lambda x : x[1],reverse = True )
	badrate = pd.DataFrame(f)
	badrate.columns = pd.Series(['cut','badrate'])
	badrate = badrate.sort_values('cut')
	print(badrate)
	badrate.plot('cut','badrate')