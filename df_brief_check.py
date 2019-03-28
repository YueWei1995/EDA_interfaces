# #########################################################################################################################
# 该函数用来快速统计一个dataframe中各个列的特征值个数、缺失值所占比例、占比最大的特征所占的比例等信息
# 输入：一个dataframe
# 输出：关于统计信息的dataframe

def df_brief_check(df):
  stats = []       # 用于保存训练集中各列的统计信息
  for col in df.columns:
    # 列名、该列的不同特征值的个数、该列的缺失值所占的比例、该列中数量最多的特征值所占的比例（包含缺失值）、该列的数据类型
    stats.append((col, df[col].nunique(), df[col].isnull().sum() * 100 / df.shape[0], df[col].value_counts(normalize=True, dropna=False).values[0] * 100, df[col].dtype))  
  stats_df = pd.DataFrame(stats, columns=['Feature', 'Unique_values', 'Percentage of missing values', 'Percentage of values in the biggest category', 'type'])
  stats_df.sort_values('Percentage of missing values', ascending=False)
  return stats_df
  
  
  
