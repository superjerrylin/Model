
import pandas as pd
import numpy as np
import datetime
import random
import time
import math
import re

### 讀取資料
df = pd.read_csv('raw_data_all.csv')

### 首先查看每個變量的格式
df.dtypes

### 查看變量的分布
df.describe()


#========================================================#
#                       時間格式處理                      #
#     start_time \ update_time \ app_date \ comm_time    #
#                                                        #
#  start_time                                            #
#    1. 先计算长度                                        #
#    2. 空格符处理                                        #
#    3. 有"费"的数据提取到新的列，原来值填充为空             #
#    4. 文本分析                                          #
#    5. 异常值处理                                        #
#    6. 时间格式统一                                      #
#    7. 缺失值填充为 2100-12-31                           #
#  update_time                                           #
#    1. 空格符处理                                        #
#    2. 异常值处理                                        #
#    3. 时间格式统一                                      #
#    4. 缺失值填充为 2100-12-31                           #
#  app_date                                              #
#    1. 空格符处理                                        #
#    2. 时间格式统一                                      #
#    3. 缺失值填充为 2100-12-31                           #
#  comm_time                                             #
#    1. 空格符处理                                        #
#    2. 计算长度                                          #
#    3. 异常值处理                                        #
#    4. 缺失值填充 -1                                     #
#========================================================#

#----------------#
#   start_time   #
#----------------#

### 删去空格字符函数
def removespace(x):
    if '<U+00A0>' in str(x):
        return str(x).replace('<U+00A0>','')
    elif '<U+FFFD>' in str(x):
        return str(x).replace('<U+FFFD>','')
    else:
        return str(x)
    
### 查询有“费”和“元”这个字，取里面数字
def fee_(x):
    if '费' in str(x['start_time']):
        return str(x['start_time']).split(":")[1].split("元")[0]
    else:
        return str(x['comm_fee'])

### 其他异常处理 时间错误  
def start_time_to_year(x):
    if '2015-2015' in str(x):
        return x.replace('2015-2015','2015')
    elif '2016-2016' in str(x):
        return x.replace('2016-2016','2016')
    elif '2016-' == str(x):
        return '2100/12/31 00:00:00'
    elif len(str(x)) == 43:
        return '2015-12-11 12:06:02'
    else:
        return x

### 长度为2的有“被叫” 这个词，对comm_mode进行填补
def start_time_to_comm_mode(x):
    if '被叫' in str(x['start_time']) and x['comm_mode'] is np.nan:
        return x['start_time']
    else:
        return x['comm_mode']

### 时间格式转换
def date_str_trans(x):
    if len(str(x)) == 19:
        try:
            date_ = datetime.datetime.strptime(x, "%Y-%m-%d %H:%M:%S")
        except:
            try:
                date_ = datetime.datetime.strptime(x, "%Y/%m/%d %H:%M:%S")
            except: 
                # 空值的时间定义为2100/12/31 00:00:00
                date_ = datetime.datetime.strptime('2100/12/31 00:00:00', "%Y/%m/%d %H:%M:%S")
    elif len(str(x)) <= 10:
        try:
            date_ = datetime.datetime.strptime(x, "%Y-%m-%d")
        except:
            try:
                date_ = datetime.datetime.strptime(x, "%Y/%m/%d")
            except: 
                # 空值的时间定义为2100/12/31 00:00:00
                date_ = datetime.datetime.strptime('2100/12/31', "%Y/%m/%d")
    else:
        data_ = datetime.datetime.strptime('2100/12/31 00:00:00', "%Y/%m/%d %H:%M:%S")
            
    return date_

### 其他异常处理 长度为18的做转换  2015-11-2212:27:20  -> 2015-11-22 12:27:20
def time_18_fun(x):
    if len(str(x)) == 18:
        return str(x)[:10] + ' ' + str(x)[10:]
    elif len(str(x)) < 4:
        return '2100/12/31 00:00:00'
    else:
        return str(x)

### 删去空格字符
df['start_time'] = df['start_time'].apply(lambda x: removespace(x))

### 有"费"的数据提取到start_time_fee列，提取完赋予空值
df['start_time_fee'] = df.apply(lambda x: fee_(x), axis = 1)
df['start_time'] = df['start_time'].apply(lambda x: '' if '费' in str(x) else str(x))
#df['comm_fee'] = df['start_time'].apply(lambda x: re.findall(r'\d+.?\d*',x)[0])

### 其他异常处理
df['start_time'] = df['start_time'].apply(lambda x: start_time_to_year(x))

### 有"被叫"的数据提取到comm_mode列，提取完赋予空值
df['comm_mode'] = df.apply(lambda x: start_time_to_comm_mode(x), axis = 1)
df['start_time'] = df['start_time'].apply(lambda x: '' if '被叫' in str(x) else str(x))

### 长度为18的做转换
df['start_time'] = df['start_time'].apply(lambda x: time_18_fun(x))

### 检查
df['start_time'].apply(lambda x:len(str(x))).value_counts()

### 查看不同长度的结果
df.loc[df['start_time'].apply(lambda x: len(str(x))) == 24, 'start_time']
  
### 修正异常值
df.loc[df['start_time'].apply(lambda x: len(str(x))) == 24, 'start_time'] = '2015/12/31 23:27:26'

### 时间统一转换成时间格式
df['start_time'] = pd.to_datetime(df['start_time'])

#-------------------#
#    update_time    #
#-------------------#

# 13位的数字串是毫秒级别的时间戳，通过下边的代码转换为表转格式：
def timenum_13_fun(x):
    if len(str(x)) == 13:
        timeNum = int(x)
        timeTemp = float(timeNum/1000)
        tupTime = time.localtime(timeTemp)
        stadardTime = time.strftime("%Y-%m-%d %H:%M:%S", tupTime)
        return stadardTime
    else:
        return x
### 数字时间转换
df['update_time'] = df['update_time'].apply(lambda x: timenum_13_fun(x))

### 时间统一转换成时间格式
df['update_time'] = pd.to_datetime(df['update_time'])


#------------------#
#     app_date     #
#------------------#
### 查看不同长度的结果
df.loc[df['app_date'].apply(lambda x: len(str(x))) == 10, 'app_date'].value_counts()

### 修正长度为3，空值赋予
df.loc[df['app_date'].apply(lambda x: len(str(x))) == 3, 'app_date'] = '2100/12/31 00:00:00'

### 时间统一转换成时间格式
df['app_date'] = pd.to_datetime(df['app_date'])


#-------------------#
#     comm_time     #
#-------------------#

### 删除空格
df['comm_time'] = df['comm_time'].apply(lambda x: removespace(x))

### 统一格式：小数点后两位
df['comm_time'] = df['comm_time'].apply(lambda x: round(float(x), 2))

### 异常值：赋值为0
df.loc[df['comm_time'].apply(lambda x: len(str(x))) > 6, 'comm_time'] = 0


#========================================================#
#                       文字處理                          #
#      comm_mode \ comm_plac \ comm_fee \ comm_type      #
#                                                        #
#  comm_mode                                             #
#    1. 计算每个值的数量                                   #
#    2. 异常值处理                                        #
#  comm_plac                                             #
#    1. 空格符处理                                        #
#    2. 与费用相关的，新增费用字段，最终再做比较             #
#    3. 字符串处理(如：特殊符号)                           #
#    4. 城市等级转换                                      #
#  comm_fee                                              #
#    1. 空格符处理                                        #
#    2. 异常值处理(如：包含非数字)                         #
#    3. 与新衍生出来的费用(start_time_fee\comm_plac_fee)   #
#       进行比较，取最大值                                 #
#  comm_type                                             #
#    1. 通话种类转换                                      #
#========================================================#
















