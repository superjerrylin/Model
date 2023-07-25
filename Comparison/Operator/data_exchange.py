
import pandas as pd
import numpy as np
from chinese_calendar import is_workday, is_holiday
import chinese_calendar as calendar
import time, datetime

data = pd.read_csv('df_clean.csv', encoding = 'gbk')
data.shape
data.head()

### 数据处理
data.loc[data['city_level'] == '本地','city_level'] = '本地地区'

### 时间差
df = data.copy()
df['app_date'] = pd.to_datetime(df['app_date'])
df['start_time'] = pd.to_datetime(df['start_time'])

df['day_diff'] = df.apply(lambda x: (x['app_date'] - x['start_time']).days - 30, axis = 1)
df = df[df['day_diff'] > 0]

### 计算时间间隔
df['day_diff_intv'] = pd.cut(df.day_diff, 
                            bins = [-np.inf] + [int(i) for i in np.arange(0,360,15)] + [np.inf], 
                            labels = [str(i) for i in np.arange(0,375,15)]
                           )
df['day_diff_intv'] = df['day_diff_intv'].astype(str)

### 新增一列
df['no'] = 1



def is_weekday(date):
    '''
    判断是否为工作日
    '''
    Y = date.year
    M = date.month
    D = date.day
    april_last = datetime.date(Y, M, D)
    if is_workday(april_last) == True:
        return 'work'
    else:
        return 'holiday'

def day_4status(x):
    '''
    判断一天的早上、下午、晚上及半夜
    '''
    hour_ = x.hour
    if hour_ in range(7,13):
        return 'morning'
    elif hour_ in range(13,19):
        return 'afternoon'
    elif hour_ in range(19,25):
        return 'night'
    else:
        return 'midnight'

df['weekday'] = df['start_time'].apply(lambda x: is_weekday(x))
df['day_status'] = df['start_time'].apply(lambda x: day_4status(x))

def phone_cnt_fn(data, var):
    
    result = pd.DataFrame()
    result['cell_phone'] = list(set(data.cell_phone))
    
    for i in var:
        tmp1 = pd.pivot_table(df, 
                               index = ['cell_phone'], 
                               columns = ['day_diff_intv',i], 
                               values = 'no', 
                               aggfunc = 'count', 
                               fill_value = 0) \
                        .reset_index()
        tmp1.columns = ['cell_phone'] + ['d'+ '_'.join(list(i)) for i in tmp1.columns.tolist()[1:]]
        result = result.merge(tmp1, on = 'cell_phone', how = 'left')
            
    return result

def phone_sum_fn(data, var):
    
    result = pd.DataFrame()
    result['cell_phone'] = list(set(data.cell_phone))
    
    col_name = ''
    
    for i in var:
        if i == 'comm_fee':
            col_name = '费用'

        if i == 'comm_time':
            col_name = '时长'
            
        tmp2 = pd.pivot_table(df, 
                               index = ['cell_phone'], 
                               columns = ['day_diff_intv'], 
                               values = i, 
                               aggfunc = 'sum', 
                               fill_value = 0) \
                        .reset_index()
        tmp2.columns = ['cell_phone'] + ['d'+ m + '_' + col_name for m in tmp2.columns.tolist()[1:]]
        result = result.merge(tmp2, on = 'cell_phone', how = 'left')
            
    return result

def phone_anoth_fn(data, var):
    
    result = data.groupby(['cell_phone','day_diff_intv'])[var].nunique().unstack(level = -1).reset_index().fillna(0)
    result.columns = ['cell_phone'] + ['d'+ str(i) + '_anoth' for i in result.columns.tolist()[1:]]
    
    return result

### 數據合併
result1 = phone_cnt_fn(df, ['comm_mode','city_level','type1','comm_type_1','weekday','day_status'])
result2 = phone_sum_fn(df, ['comm_time','comm_fee'])
result3 = phone_anoth_fn(df, 'another_nm')
result = pd.merge(result1, result2, on = 'cell_phone', how = 'left')
result = result.merge(result3, on = 'cell_phone', how = 'left')
result.shape

### 資料轉換
result_final = pd.DataFrame()
result_final['cell_phone'] = result.cell_phone.copy()
result_var = result.columns.tolist()

for i in day_list:
    for j in var_list:
        
        var_name = 'd' + str(i) + '_' + j
        if var_name in result_var:
            result_final[var_name] = result[var_name]
        else:
            result_final[var_name] = 0
### 保存資料
result_final.to_csv('yys_lstm_data.csv', encoding = 'gbk', index = False)


