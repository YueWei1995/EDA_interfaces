
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
  
  
