
# 数据类型转换
train['Census_OSBuildRevision'] = train['Census_OSBuildRevision'].astype('category')
# ##################################################################################################

# 把一个字符串特征中的字母全部转为小写，然后归并特征值
def rename_edition(x):
    x = x.lower()
    if 'core' in x:
        return 'Core'
    elif 'pro' in x:
        return 'pro'
    elif 'cloud' in x:
        return 'Cloud'
    else:
        return x
# ##################################################################################################

# 另一种特征值的方法，这里对train这个dataframe中的"SmartScreen"列进行归并
trans_dict = {
    'off': 'Off', '&#x02;': '2', '&#x01;': '1', 'on': 'On', 'requireadmin': 'RequireAdmin', 'OFF': 'Off', 
    'Promt': 'Prompt', 'requireAdmin': 'RequireAdmin', 'prompt': 'Prompt', 'warn': 'Warn', 
    '00000000': '0', '&#x03;': '3', np.nan: 'NoExist'
}
train.replace({'SmartScreen': trans_dict}, inplace=True)
# ##################################################################################################

# 对于一个分类特征，只保留取值数量最多的前10个特征值，将剩余取值过少的特征值合并为一个特征值
top_10 = train['Census_TotalPhysicalRAM'].value_counts(dropna=False, normalize=True).cumsum().index[:10]
train.loc[train['Census_TotalPhysicalRAM'].isin(top_10) == False, 'Census_TotalPhysicalRAM'] = 1000
test.loc[test['Census_TotalPhysicalRAM'].isin(top_10) == False, 'Census_TotalPhysicalRAM'] = 1000    
# ##################################################################################################
      
# 找到train中所有数据类型为category的列，即分类特征；然后调用labelEncoder对这些列进行编码
cate_cols = train.select_dtypes(include='category').columns.tolist()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for col in cate_cols:
    train[col] = le.fit_transform(train[col])
# ##################################################################################################

# 根据一个字典的映射关系，从一个dataframe的一列中产生新的一列特征
datedict = np.load('../input/malware-timestamps/AvSigVersionTimestamps.npy')
datedict = datedict[()]
df_train['Date'] = df_train['AvSigVersion'].map(datedict)
df_test['Date'] = df_test['AvSigVersion'].map(datedict)
# ##################################################################################################




