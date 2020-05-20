
def feature_select_cor_iv(cormat, iv_df, cor_thred, iv_thred): 
	'''
	执行步骤：
		(1)剔除IV值<iv_thred的特征，剔除特征放入列表except_var，并对iv_df按IV值降序排列；
	    (2)按IV值从大到小选择iv_df中特征Var，若特征Var不在except_var中，则将Var放入列表remain_var；
	    (3)剔除与Var相关系数大于cor_thred的特征，并将剔除特征加入列表except_var；
	    (4)重复步骤(2)和(3)，直至结束，则列表remain_var中的特征即为选择的特征。
    cormat是相关系数数据框
    iv_df是IV值数据框
    cor_thred是相关系数的阈值
    iv_thred是IV值阈值。
	'''
	except_var = set(iv_df['Var'][(~iv_df['Var'].isin(cormat.index)) | (iv_df['Iv']<iv_thred)]) 
	iv_df = iv_df.sort_values(by=['Iv'], ascending=[False]) 
	remain_var = [] 
	for var in iv_df.Var: 
		if var in except_var: 
			continue 
		else: 
			corr = cormat.loc[(cormat[var]>cor_thred) & (cormat[var]<1), var] 
		except_var = except_var.union(set(corr.index)) 
		remain_var.append(var) 
		except_var = list(except_var) 
	return remain_var, except_var

#样例
#remain_vars, ex_vars = Feature_Select_cor_iv(corr_mat, iv_df, 0.7, 0.03)