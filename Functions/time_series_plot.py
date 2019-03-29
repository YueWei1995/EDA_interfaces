# 来源自：https://www.kaggle.com/cdeotte/time-series-eda-malware-0-64/notebook

import calendar, math

# 这两个函数的代码可读性太差了，不知道在写些什么东西
# two EDA Python functions. 
# One function visualizes overall density and detection rate per category value. 可视化每个分类特征的整体密度和正例率
# The other visualizes density and detection rate over time.  可视化样本随时间的密度和正例率
# ############################################################################################################################
# PARAMETERS
# data : pandas.DataFrame : your data to plot     要绘制的数据，是一个dataframe
# col  : str : which column to plot histogram for left y-axis       选择一列作为要绘制的直方图的左y轴
# target : str : which column for mean/rate on right y-axis         选择右y轴上的显示要比率的列，应该是标签label列
# bars : int : how many histogram bars to show (or less if you set show or min)       要显示的直方的数量
# show : float : stop displaying bars after 100*show% of data is showing              如果已绘制的数据超过总数据的100*show%，就停止绘制
# minn : float : don't display bars containing under 100*minn% of data                如果一个直方所包含的数据小于总数据的100*minn%，就不绘制它
# sortby : str : either 'frequency', 'category', or 'rate'                  可选'frequency'\'category'\'rate'，分别为根据‘频次’、‘类别’、‘比率’排序
# verbose : int : display text summary 1=yes, 0=no                          显示文本摘要，1为是，0为否
# top : int : give this many bars nice color (and matches a subsequent dynamicPlot)   给top个直方添加漂亮的颜色（并匹配后续的动态图）
# title : str : title of plot                                               图的标题
# asc : boolean : sort ascending (for category and rate)                    是否按升序排列
# dropna : boolean : include missing data as a category or not              是否将缺失值作为一个特征值，即是否舍弃缺失值
# 调用实例：
  # staticPlot(df_train,'SmartScreen',title='SmartScreen')
  # dynamicPlot(df_train,'SmartScreen',title='SmartScreen')

def staticPlot(data, col, target, bars=10, show=1.0, sortby='frequency'
               , verbose=1, top=5, title='',asc=False, dropna=False, minn=0.0):
    # calcuate density and detection rate       计算密度和正例率
    cv = data[col].value_counts(dropna=dropna)
    cvd = cv.to_dict()
    nm = cv.index.values; lnn = len(nm); lnn2 = lnn
    th = show * len(data)
    th2 = minn * len(data)
    sum = 0; lnn2 = 0
    for x in nm[0:bars]:
        lnn2 += 1
        try: sum += cvd[x]
        except: sum += cv[x]
        if sum>th:
            break
        try:
            if cvd[x]<th2: break
        except:
            if cv[x]<th2: break
    if lnn2<bars: bars = lnn2
    pct = round(100.0*sum/len(data),2)
    lnn = min(lnn,lnn2)
    ratio = [0.0]*lnn; lnn3 = lnn
    if sortby =='frequency': lnn3 = min(lnn3,bars)
    elif sortby=='category': lnn3 = 0
    for i in range(lnn3):
        if target not in data:
            ratio[i] = np.nan
        elif nan_check(nm[i]):
            ratio[i] = data[target][data[col].isna()].mean()
        else:
            ratio[i] = data[target][data[col]==nm[i]].mean()
    try: all = pd.DataFrame( {'category':nm[0:lnn],'frequency':[cvd[x] for x in nm[0:lnn]],'rate':ratio} )
    except: all = pd.DataFrame( {'category':nm[0:lnn],'frequency':[cv[x] for x in nm[0:lnn]],'rate':ratio} )
    if sortby=='rate': 
        all = all.sort_values(sortby, ascending=asc)
    elif sortby=='category':
        try: 
            all['temp'] = all['category'].astype('float')
            all = all.sort_values('temp', ascending=asc)
        except:
            all = all.sort_values('category', ascending=asc)
    if bars<lnn: all = all[0:bars]
    if verbose==1 and target in data:
        print('TRAIN.CSV variable',col,'has',len(nm),'categories')
        print('The',min(bars,lnn),'bars displayed here contain',pct,'% of data.')
        mlnn = data[col].isna().sum()
        print("The data has %.1f %% NA. The plot is sorted by " % (100.0*mlnn/len(data)) + sortby )
    
    # plot density and detection rate
    fig = plt.figure(1,figsize=(15,3))
    ax1 = fig.add_subplot(1,1,1)
    clrs = ['red', 'green', 'blue', 'yellow', 'magenta']
    barss = ax1.bar([str(x) for x in all['category']],[x/float(len(data)) for x in all['frequency']],color=clrs)
    for i in range(len(all)-top):
        barss[top+i].set_color('cyan')
    if target in data:
        ax2 = ax1.twinx()
        if sortby!='category': infected = all['rate'][0:lnn]
        else:
            infected=[]
            for x in all['category']:
                if nan_check(x): infected.append( data[ data[col].isna() ][target].mean() )
                elif cvd[x]!=0: infected.append( data[ data[col]==x ][target].mean() )
                else: infected.append(-1)
        ax2.plot([str(x) for x in all['category']],infected[0:lnn],'k:o')
        #ax2.set_ylim(a,b)
        ax2.spines['left'].set_color('red')
        ax2.set_ylabel('Detection Rate', color='k')
    ax1.spines['left'].set_color('red')
    ax1.yaxis.label.set_color('red')
    ax1.tick_params(axis='y', colors='red')
    ax1.set_ylabel('Category Proportion', color='r')
    if title!='': plt.title(title)
    plt.show()
    if verbose==1 and target not in data:
        print('TEST.CSV variable',col,'has',len(nm),'categories')
        print('The',min(bars,lnn),'bars displayed here contain',pct,'% of the data.')
        mlnn = data[col].isna().sum()
        print("The data has %.1f %% NA. The plot is sorted by " % (100.0*mlnn/len(data)) + sortby )
        
        
# #########################################################################################################################
# PARAMETERS
# data : pandas.DataFrame : your data to plot                             要绘制的数据集的dataframe
# col  : str : which column for density on left y-axis                    选择一列作为左y轴的密度
# target : str : which column for mean/rate on right y-axis               选择一列作为右y轴的正例率，即label列
# start : datetime.datetime : x-axis minimum                              起始时间，x轴的最小值
# end : datetime.datetime : x-axis maximum                                终止时间，x轴的最大值
# inc_hr : int : resolution of time sampling = inc_hr + inc_dy*24 + inc_mn*720 hours          
# inc_dy : int : resolution of time sampling = inc_hr + inc_dy*24 + inc_mn*720 hours
# inc_mn : int : resolution of time sampling = inc_hr + inc_dy*24 + inc_mn*720 hours
# show : float : only show the most frequent category values that include 100*show% of data           只显示包含所有数据的100*show%以上的最频繁的分类特征值
# top : int : plot this many solid lines                      绘制top条实线
# top2 : int : plot this many dotted lines                    绘制top2条虚线
# title : str : title of plot                                 图的标题
# legend : int : include legend or not. 1=yes, 0=no           是否包含legend（图例说明），1为是，0为否
# dropna : boolean : include missing data as a category or not    是否将缺失值作为一个特征值，即是否舍弃缺失值
# 调用实例：
  # df_train['ones'] = 1
  # dynamicPlot(df_train,'ones',title='Train Data Density versus Time',legend=0)
        
def dynamicPlot(data,col, target='HasDetections', start, end=datetime(2
                ,inc_hr=0,inc_dy=7,inc_mn=0,show=0.99,top=5,top2=4,title='',legend=1, dropna=False):
    # check for timestamps
    if 'Date' not in data:
        print('Error dynamicPlot: DataFrame needs column Date of datetimes')
        return
    
    # remove detection line if category density is too small
    cv = data[col].value_counts(dropna=dropna)
    cvd = cv.to_dict()
    nm = cv.index.values
    th = show * len(data)
    sum = 0; lnn2 = 0
    for x in nm:
        lnn2 += 1
        try: sum += cvd[x]
        except: sum += cv[x]
        if sum>th:
            break
    top = min(top,len(nm))
    top2 = min(top2,len(nm),lnn2,top)

    # calculate rate within each time interval
    diff = (end-start).days*24*3600 + (end-start).seconds
    size = diff//(3600*((inc_mn * 28 + inc_dy) * 24 + inc_hr)) + 5
    data_counts = np.zeros([size,2*top+1],dtype=float)
    idx=0; idx2 = {}
    for i in range(top):
        idx2[nm[i]] = i+1
    low = start
    high = add_time(start,inc_mn,inc_dy,inc_hr)
    data_times = [low+(high-low)/2]
    while low<end:
        slice = data[ (data['Date']<high) & (data['Date']>=low) ]
        data_counts[idx,0] = len(slice)
        for key in idx2:
            if nan_check(key): slice2 = slice[slice[col].isna()]
            else: slice2 = slice[slice[col]==key]
            data_counts[idx,idx2[key]] = len(slice2)
            if target in data:
                data_counts[idx,top+idx2[key]] = slice2['HasDetections'].mean()
        low = high
        high = add_time(high,inc_mn,inc_dy,inc_hr)
        data_times.append(low+(high-low)/2)
        idx += 1

    # plot lines
    fig = plt.figure(1,figsize=(15,3))
    cl = ['r','g','b','y','m']
    ax3 = fig.add_subplot(1,1,1)
    lines = []; labels = []
    for i in range(top):
        tmp, = ax3.plot(data_times,data_counts[0:idx+1,i+1],cl[i%5])
        lines.append(tmp)
        labels.append(str(nm[i]))
    ax3.spines['left'].set_color('red')
    ax3.yaxis.label.set_color('red')
    ax3.tick_params(axis='y', colors='red')
    if col!='ones': ax3.set_ylabel('Category Density', color='r')
    else: ax3.set_ylabel('Data Density', color='r')
    ax3.set_yticklabels([])
    if target in data:
        ax4 = ax3.twinx()
        for i in range(top2):
            ax4.plot(data_times,data_counts[0:idx+1,i+1+top],cl[i%5]+":")
        ax4.spines['left'].set_color('red')
        ax4.set_ylabel('Detection Rate', color='k')
    if title!='': plt.title(title)
    if legend==1: plt.legend(lines,labels)
    plt.show()

# ################################################################################################################
# 以下是两个工具函数
# INCREMENT A DATETIME
def add_time(sdate,months=0,days=0,hours=0):
    month = sdate.month -1 + months
    year = sdate.year + month // 12
    month = month % 12 + 1
    day = sdate.day + days
    if day>calendar.monthrange(year,month)[1]:
        day -= calendar.monthrange(year,month)[1]
        month += 1
        if month>12:
            month = 1
            year += 1
    hour = sdate.hour + hours
    if hour>23:
        hour = 0
        day += 1
        if day>calendar.monthrange(year,month)[1]:
            day -= calendar.monthrange(year,month)[1]
            month += 1
            if month>12:
                month = 1
                year += 1
    return datetime(year,month,day,hour,sdate.minute)

# ###############################################################################################################
# CHECK FOR NAN
def nan_check(x):
    if isinstance(x,float):
        if math.isnan(x):
            return True
    return False

