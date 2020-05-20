'''
基于浅层决策树进行自动化的变量组合，常用于评分卡模型中的特征交叉，以及风控策略规则的自动化挖掘。
通常的策略分析使用单变量的IV（Information Value）以及其分箱后的负样本占比（badrate）进行变量挑选以及阈值确定。
单特征之间相互独立的分析并不能保证其组合后的结果。而决策树可以通过贪心的思路进行规则组合
'''
import toad
import pandas as pd
import numpy as np
import pydotplus
from IPython.display import Image
from sklearn.externals.six import StringIO
import os
from sklearn import tree


class auto_policy(object):

    def __init__(self, datasets, ex_lis, dep='bad_ind', min_samples=0.05, min_samples_leaf=200, min_samples_split=20,
                 max_depth=4, is_bin=True):

        '''
        datasets:数据集 dataframe格式
        ex_lis：不参与建模的特征，如id，时间切片等。 list格式
        min_samples：分箱时最小箱的样本占总比 numeric格式
        max_depth：决策树最大深度 numeric格式
        min_samples_leaf：决策树子节点最小样本个数 numeric格式
        min_samples_split：决策树划分前，父节点最小样本个数 numeric格式
        is_bin：是否进行卡方分箱 bool格式（True/False）
        '''
        self.datasets = datasets
        self.ex_lis = ex_lis
        self.dep = dep
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.is_bin = is_bin

        self.bins = 0

    def fit_plot(self):
        os.environ["PATH"] += os.pathsep + 'D:/Program Files/Graphviz2.38/bin'
        dtree = tree.DecisionTreeRegressor(max_depth=self.max_depth,
                                           min_samples_leaf=self.min_samples_leaf,
                                           min_samples_split=self.min_samples_split)

        x = self.datasets.drop(self.ex_lis, axis=1)
        y = self.datasets[self.dep]

        if self.is_bin:
            # 分箱
            combiner = toad.transform.Combiner()
            combiner.fit(x, y, method='chi', min_samples=self.min_samples)

            x_bin = combiner.transform(x)
            self.bins = combiner.export()
        else:
            x_bin = x.copy()

        dtree = dtree.fit(x_bin, y)

        df_bin = x_bin.copy()

        df_bin[self.dep] = y

        dot_data = StringIO()
        tree.export_graphviz(dtree, out_file=dot_data,
                             feature_names=x_bin.columns,
                             class_names=[self.dep],
                             filled=True, rounded=True,
                             special_characters=True)
        graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
        '''
        df_bin 数据，分箱后用箱编号代替原变量值
        bins 分箱详情，可找到每个分箱的具体逻辑
        combiner 分箱的工具，通过combiner.transform可以将其他数据文件分箱
        graph 图像文件，需要通过Image函数打印出来
        '''

        return df_bin, self.bins, combiner, graph.create_png()



#读取数据
rh_base = pd.read_excel(path + 'rh_base.xlsx')
#指定不参与建模的变量，包含标签bad_ind。
ex_lis = ['id','bad_ind','dt']
#调用决策树函数
df_bin,bins,combiner,graph = auto_policy(datasets = rh_base, ex_lis = ex_lis,
                                         dep = 'bad_ind', min_samples=0.01,
                                         min_samples_leaf=50, min_samples_split=50).fit_plot()
#展示图像
Image(graph)

#查看数据集中单个变量的分箱结果
set(df_bin['maritalstate'])

#查看单个变量的分箱详情
bins['maritalstate']