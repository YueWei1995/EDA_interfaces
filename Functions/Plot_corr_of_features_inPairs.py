
import seaborn as sns
import matplotlib.pyplot as plt
# ##########################################################################
# 绘制数个特征间的两两的相关性的热度图，以便直观地判断特征间的相关度
# 输入：df为数据集的dataframe；cols为特征名（列名）的列表，如['A', 'B', 'C', 'D'];label为数据集的标签名
# 输出：相关性热度图
def Plot_corr_of_features_inPairs(df, cols, label):
  plt.figure(figsize=(10,10))
  cols.append(label)
  sns.heatmap(df[cols].corr(), cmap='RdBu_r', annot=True, center=0.0)
  plt.title('Correlation between {} columns'.format(len(cols))
  plt.show()
  
# ##########################################################################
# 与上一个函数类似，绘制一个dataframe中所有特征的两两间的相关度的热度图，相关度小于0.99的用0表示，大于0.99的用1表示，
# 输入：一个dataframe
# 输出：所有特征的两两间的相关性热度图
def Plot_corr_of_features_inPairs_1(df)
  corr = df.corr()
  high_corr = (corr >= 0.99).astype('uint8')
  plt.figure(figsize=(15,15))
  sns.heatmap(high_corr, cmap='RdBu_r', annot=True, center=0.0)
  plt.show()           
