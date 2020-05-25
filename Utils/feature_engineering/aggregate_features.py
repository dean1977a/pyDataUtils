def aggregate_features(df_, prefix):

    df = df_.copy()

    agg_func = {
        '特征1':  ['count','nunique'],
        '特征2':  ['nunique'],
        '特征3':  ['mean','max','min','std'],
        } 

    agg_df = df.groupby(['主键']).agg(agg_func)
    agg_df.columns = [prefix + '_'.join(col).strip() for col in agg_df.columns.values]
    agg_df.reset_index(drop=False, inplace=True)
    
    return agg_df