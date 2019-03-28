
# 数据类型转换
train['Census_OSBuildRevision'] = train['Census_OSBuildRevision'].astype('category')

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

# 对于一个分类特征，只保留取值数量最多的前10个特征值，将剩余取值过少的特征值合并为一个特征值
top_10 = train['Census_TotalPhysicalRAM'].value_counts(dropna=False, normalize=True).cumsum().index[:10]
train.loc[train['Census_TotalPhysicalRAM'].isin(top_10) == False, 'Census_TotalPhysicalRAM'] = 1000
test.loc[test['Census_TotalPhysicalRAM'].isin(top_10) == False, 'Census_TotalPhysicalRAM'] = 1000     
      
