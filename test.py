# _*_ coding: utf-8 _*_

import pandas as pd
import numpy as np
from utils import reduce_mem_usage
import plotly.offline as py
import plotly.graph_objs as go


# 绘制一个数据集中一个分类特征在label(0或1)上的分布
# 输入: df为数据集dataframe、cate_col为分类特征名、label_col为标签名、top_n为只统计分类特征的前n个特征值，only_bars指示是否画出折线图
# 输出： 一个html格式的图片文件，
def plot_categorical_feature(df, cate_col, label_col, top_n=10, only_bars=False):
    top_n = top_n if df[cate_col].nunique() > top_n else df[cate_col].nunique()
    print('{} has {} unique values'.format(cate_col, df[cate_col].nunique()))
    if not only_bars:
        result_df = df.groupby([cate_col]).agg({label_col: ['count', 'mean']})
        result_df = result_df.sort_values((label_col, 'count'), ascending=False).head(top_n).sort_index()
        data = [go.Bar(x=result_df.index, y=result_df[label_col]['count'].values, name='counts'),
                go.Scatter(x=result_df.index, y=result_df[label_col]['mean'], name='ratio', yaxis='y2')]

        layout = go.Layout(dict(title='Counts of {} by top-{} categories and mean target value'.format(cate_col, top_n),
                                xaxis=dict(title='{}'.format(cate_col),
                                           showgrid=False,
                                           zeroline=False,
                                           showline=False,),
                                yaxis=dict(title='Counts',
                                           showgrid=False,
                                           zeroline=False,
                                           showline=False,),
                                yaxis2=dict(title='Ratio', overlaying='y', side='right')),
                           legend=dict(orientation='v'))
    else:
        top_cate = list(df[cate_col].value_counts(dropna=False).index[:top_n])
        result_df0 = df.loc[(df[cate_col].isin(top_cate)) & (df[label_col] == 1), cate_col].value_counts().head(10).sort_index()
        result_df1 = df.loc[(df[cate_col].isin(top_cate)) & (df[label_col] == 0), cate_col].value_counts().head(10).sort_index()
        data = [go.Bar(x=result_df0.index, y=result_df0.values, name='positive'),
                go.Bar(x=result_df1.index, y=result_df1.values, name='negative')]

        layout = go.Layout(dict(title='Counts of {} by top-{} categories'.format(cate_col, top_n),
                                xaxis=dict(title='{}'.format(cate_col),
                                           showgrid=False,
                                           zeroline=False,
                                           showline=False,),
                                yaxis=dict(title='Counts',
                                           showgrid=False,
                                           zeroline=False,
                                           showline=False,),
                                ),
                           legend=dict(orientation='v'), barmode='group')
    py.plot(dict(data=data, layout=layout))


file_df = pd.read_table('../data/train.txt', sep='\t')
columns = ['uid', 'user_city', 'item_id', 'author_id', 'item_city', 'channel', 'finish', 'like', 'music_id',
           'device_id', 'creat_time', 'video_duration']
file_df.columns = columns
reduce_mem_usage(file_df)

plot_categorical_feature(file_df, 'channel', 'finish', 10, True)

















