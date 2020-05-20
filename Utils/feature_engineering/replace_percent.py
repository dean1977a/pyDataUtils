def replace_percent(df,cols):
	'''
    对数据集中含有%的列进行替换

	'''
	for col in cols:
		df[col+'_clean'] = df[col].map(lambda x: float(x.replace('%', '')) / 100)

	return df