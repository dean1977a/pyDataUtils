#https://blog.csdn.net/CherDW/article/details/102788196
# 首先计算时间差
def month_sub(d1, d2):
    year = int(d1[:4]) - int(d2[:4])
    month = int(d1[5:7]) - int(d2[5:7])
    return year*12 + month
data['month_diff'] = data.apply(lambda row: month_sub(row['dt'], row['billdate']), axis=1)


def feature(data,cols:list,months:list,only_mark,merge_ori=True):
    start = time.time()
    # 原数据集的 唯一主键
    df = pd.DataFrame({only_mark:list(set(data[only_mark]))})
    for month in months:
        df1 = pd.DataFrame({only_mark:list(set(data[only_mark]))})
        for col in cols:
            agg_dict = {
                        "last_%s_%s_count"%(month,col):"count",
                        "last_%s_%s_sum"%(month,col):"sum",
                        "last_%s_%s_max"%(month,col):"max",
                        "last_%s_%s_min"%(month,col):"min",
                        "last_%s_%s_mean"%(month,col):"mean",
                        "last_%s_%s_var"%(month,col):"var",
                        "last_%s_%s_std"%(month,col):"std",
                        "last_%s_%s_median"%(month,col):"median",
                        "last_%s_%s_skew"%(month,col):"skew"
                                                                }
            # 选取时间切片内的数据 进行groupby聚合计算
            sta_data = data[data['month_diff']<=month].groupby([only_mark])[col].agg(agg_dict).reset_index()
            df1 = df1.merge(sta_data,how = "left",on = only_mark)
        df = df.merge(df1,how = "left",on = only_mark)
    # 是否与原数据集关联  
    if merge_ori:
        print("merge the original data")
        df = df.merge(data,how="right",on=only_mark).fillna(0) #视情况定 是否需要填充0
    else:
        df = df.fillna(0)
    end = time.time()
    cost = end-start
    print("cost time %.2f s"%(cost))
    return  df
if __name__=="__main__":
    cols = ['amount', 'interest']
    months = [1,3]
    result=feature(data,cols,months,"id",merge_ori = 0)