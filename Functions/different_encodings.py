# 来源自：https://www.kaggle.com/cdeotte/time-series-eda-malware-0-64/notebook
# 多种不同的特征编码方式：标签编码、频次编码、二元编码、数值编码

# If you call encode_TE, encode_TE_partial, encode_FE_partial, 
# or encode_BE_partial on training data then the function 
# returns a 2 element python list containing [list, dictionary]
# the return[0] = list are the names of new columns added
# the return[1] = dictionary are which category variables got encoded
# When encoding test data after one of 4 calls above, use 'encode_?E_test'
# and pass the dictionary. If you don't use one of 4 above, then you can
# call basic 'encode_?E' on test.
# 如果你在训练数据上调用encode_TE, encode_TE_partial, encode_FE_partial, 或encode_BE_partial，函数会返回一个含两个元素[list, dictionary]的Python列表,
# return[0] = list是新增的列的列名；return[1] = dictionary是哪些分类特征被编码了
# 当在上述4个调用之后对测试数据进行编码时，使用'encode_?E_test'并传入字典。
# 如果没有使用上述4个调用之一，则在测试集上调用基本的'encode_?E'

# TARGET ENCODING  标签编码
def encode_TE(df,col,tar):
    d = {}
    v = df[col].unique()
    for x in v:
        if nan_check(x):
            m = df[tar][df[col].isna()].mean()
        else:
            m = df[tar][df[col]==x].mean()
        d[x] = m
    n = col+"_TE"
    df[n] = df[col].map(d)
    return [[n],d]

# TARGET ENCODING first ct columns by freq
def encode_TE_partial(df,col,ct,tar,xx=0.5):
    d = {}
    cv = df[col].value_counts(dropna=False)
    nm = cv.index.values[0:ct]
    for x in nm:
        if nan_check(x):
            m = df[tar][df[col].isna()].mean()
        else:
            m = df[tar][df[col]==x].mean()
        d[x] = m
    n = col+"_TE"
    df[n] = df[col].map(d).fillna(xx)
    return [[n],d]

# TARGET ENCODING from dictionary
def encode_TE_test(df,col,mp,xx=0.5):
    n = col+"_TE"
    df[n] = df[col].map(mp).fillna(xx)
    return [[n],0]

# FREQUENCY ENCODING    频次编码
def encode_FE(df,col):
    d = df[col].value_counts(dropna=False)
    n = col+"_FE"
    df[n] = df[col].map(d)/d.max()
    return [[n],d]

# FREQUENCY ENCODING first ct columns by freq
def encode_FE_partial(df,col,ct):
    cv = df[col].value_counts(dropna=False)
    nm = cv.index.values[0:ct]
    n = col+"_FE"
    df[n] = df[col].map(cv)
    df.loc[~df[col].isin(nm),n] = np.mean(cv.values)
    df[n] = df[n] / max(cv.values)
    d = {}
    for x in nm: d[x] = cv[x]
    return [[n],d]

# FREQUENCY ENCODING from dictionary
def encode_FE_test(df,col,mp,xx=1.0):
    cv = df[col].value_counts(dropna=False)
    n = col+"_FE"
    df[n] = df[col].map(cv)
    df.loc[~df[col].isin(mp),n] = xx*np.mean(cv.values)
    df[n] = df[n] / max(cv.values)
    return [[n],mp]

# BINARY ENCODING       二元编码
def encode_BE(df,col,val='xyz'):
    if val=='xyz':
        print('BE_encoding all')
        v = df[col].unique()
        n = []
        for x in v: n.append(encode_BE(df,col,x)[0][0])
        return [n,0]
    n = col+"_BE_"+str(val)
    if nan_check(val):
        df[n] = df[col].isna()
    elif isinstance(val, (list,)):
        if not isinstance(val[0], str):
            print('BE_encode Warning: val list not str')
        n = col+"_BE_"+str(val[0])+"_"+str(val[-1])
        d = {}
        for x in val: d[x]=1
        df[n] = df[col].map(d).fillna(0)
    else:
        if not isinstance(val, str):
            print('BE_encode Warning: val is not str')
        df[n] = df[col]==val
    df[n] = df[n].astype('int8')
    return [[n],0]

# BINARY ENCODING first ct columns by freq
def encode_BE_partial(df,col,ct):
    cv = df[col].value_counts(dropna=False)
    nm = cv.index.values[0:ct]
    d = {}
    n = []
    for x in nm: 
        n.append(encode_BE(df,col,x)[0][0])
        d[x] = 1
    return [n,d]

# BINARY ENCODING from dictionary
def encode_BE_test(df,col,mp):
    n = []
    for x in mp: n.append(encode_BE(df,col,x)[0][0])
    return [n,0]

# NUMERIC ENCODING          数值编码
def encode_NE(df,col):
    n = col+"_NE"
    df[n] = df[col].astype(float)
    mx = np.std(df[n])
    mn = df[n].mean()
    df[n] = (df[n].fillna(mn) - mn) / mx
    return [[n],[mn,mx]]

# NUMERIC ENCODING from mean and std
def encode_NE_test(df,col,mm):
    n = col+"_NE"
    df[n] = df[col].astype(float)
    df[n] = (df[n].fillna(df[n].mean()) - mm[0]) / mm[1]
    return [[n],mm]
