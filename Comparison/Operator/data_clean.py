
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
#      type1 \ type2                                     #
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


#-------------------#
#     comm_plac     #
#-------------------#

### 删除空格
def removespace(x):
    if '<U+00A0>' in str(x):
        return str(x).replace('<U+00A0>','')
    elif '<U+FFFD>' in str(x):
        return str(x).replace('<U+FFFD>','')
    else:
        return x

### 查询有“费”和“元”这个字，取数字
def plac_fee(x):
    if '费' in str(x['comm_plac']):
        return x['comm_plac'].split(":")[1].split("元")[0]
    else:
        return x['comm_fee']

### 只保留汉字
def plac_ch(x):
    
    return re.sub("[A-Za-z0-9_.!+-=——,$%^，（。）；？、~@?#￥%……&*《》<>「」{}【】()/\\\[\]'\"]","", str(x))  # 删除英文、数字、符号
    #sen_text = re.compile(u'[\u4E00-\u9FA5|\s\w]').findall(str(x['comm_plac']))
    #return "".join(sen_text)

### 城市转换成一线、新一线、二线、三线及以下城市、其他地区
fst_plac = ['北京','上海','广州','深圳']
newfst_plac = ['成都','杭州','重庆','武汉','西安',
               '苏州','天津','南京','长沙','郑州',
               '东莞','青岛','沈阳','宁波','佛山']
snd_plac = ['合肥','昆明','无锡','厦门','济南','福州','温州','大连','长春','泉州',
            '石家庄','南宁','金华','哈尔滨','贵阳','南昌','常州','嘉兴','珠海','惠州',
            '中山','南通','太原','徐州','绍兴','台州','烟台','兰州','潍坊','临沂']

other_plac = ['香港','澳门','台湾','港澳台','国际','韩国','日本','芬兰','罗马尼亚']

def trans_plac(x):
    if x == '本地':
        return '本地地区'
    elif x in other_plac:   # 其他地区
        return 'other_plac'
    elif x in snd_plac:     # 二线城市
        return '2nd_plac'
    elif x in newfst_plac:  # 新一线城市
        return 'new1st_plac'
    elif x in fst_plac:     # 一线城市
        return '1st_plac'  
    else:                   # 三线及以下城市
        return '3rd_plac'
        
### 删除空格
df['comm_plac'] = df['comm_plac'].apply(lambda x: removespace(x))

### 有"费"的数据提取到start_time_fee列，提取完赋予空值
df['comm_plac_fee'] = df.apply(lambda x: plac_fee(x), axis = 1)
df['comm_plac'] = df['comm_plac'].apply(lambda x: '' if '费' in str(x) else x)

### 提取汉字
df['comm_plac'] = df.apply(lambda x: plac_ch(x['comm_plac']), axis = 1)

### 城市等级
df['city_level'] = df.apply(lambda x: trans_plac(x['comm_plac']), axis = 1)


#------------------#
#     comm_fee     #
#------------------#
### 删除空格
def removespace(x):
    if '<U+00A0>' in str(x):
        return str(x).replace('<U+00A0>','')
    elif '<U+FFFD>' in str(x):
        return str(x).replace('<U+FFFD>','')
    else:
        return x

### 删除“元”、“分”、“角”
def fee_outlier(x):
    
    # string 转 list
    ch_last = list(str(x))[-1]
    
    if '元' in ch_last:
        return re.sub('元', '', str(x))
    elif '分' in ch_last:
        dollar_1 = re.sub('分', '', str(x))
        return str(float(dollar_1)/100)
    elif '角' in ch_last:
        dollar_2 = re.sub('角', '', str(x))
        return str(float(dollar_2)/10)
    else:
        return str(x)

### 异常值处理：包含大量中文，与金钱无相关
def fee_ch_outlier(x):
    try:
        return float(x)
    except:
        return -1
    
### 取comm_fee、comm_plac_fee、start_time_fee的最大值
def fee_max(x):
    fee_list = []
    
    plac_fee_cl1 = removespace(x['comm_plac_fee'])
    plac_fee_cl2 = fee_outlier(plac_fee_cl1)
    plac_fee_cl3 = fee_ch_outlier(plac_fee_cl2)
    
    time_fee_cl1 = removespace(x['start_time_fee'])
    time_fee_cl2 = fee_outlier(time_fee_cl1)
    time_fee_cl3 = fee_ch_outlier(time_fee_cl2)
    
    comm_fee1 = fee_ch_outlier(x['comm_fee'])
    
    fee_list.append(comm_fee1)
    fee_list.append(float(plac_fee_cl3))
    fee_list.append(float(time_fee_cl3))
    
    return max(fee_list)

### 删除空格
df['comm_fee'] = df['comm_fee'].apply(lambda x: removespace(x))

### 删除“元”、“分”、“角”
df['comm_fee'] = df.apply(lambda x: fee_outlier(x['comm_fee']), axis = 1)

### 异常值处理：包含大量中文，与金钱无相关
df['comm_fee'] = df['comm_fee'].apply(lambda x: fee_ch_outlier(x))

### 取comm_fee、comm_plac_fee、start_time_fee的最大值
df['comm_fee'] = df.apply(lambda x: fee_max(x), axis = 1)

#-------------------#
#     comm_type     #
#-------------------#
### 通话种类转换

def type_group(x):
    if '国内漫游' in str(x):
        return '国内漫游'
    elif '国内长途' in str(x):
        return '国内长途'
    elif '省内长途' in str(x):
        return '省内长途'
    elif '省际长途' in str(x):
        return '省际长途'
    elif '省内漫游' in str(x):
        return '省内漫游'
    elif '省际漫游' in str(x):
        return '省际漫游'
    elif '非漫游' in str(x):
        return '非漫游'
    elif '本地市话' in str(x):
        return '本地市话'
    elif 'VPMN' in str(x):
        return 'VPMN'
    elif '本地(非漫游、被叫)' in str(x):
        return '本地(非漫游、被叫)'
    elif '本地市话' in str(x):
        return '本地市话'
    elif '市话' in str(x):
        return '市话'
    elif '国内' in str(x):
        return '国内'
    elif '普通语音' in str(x):
        return '普通语音'
    elif '省内' in str(x):
        return '省内'
    elif '省际' in str(x):
        return '省际'
    elif '漫游' in str(x):
        return '漫游'
    elif '本地' in str(x):
        return '本地'
    elif '长途' in str(x):
        return '长途'
    else:
        return ''

df['comm_type_1'] = df.apply(lambda x: type_group(x['comm_type']), axis = 1)

#-----------------#
#      type2      #
#-----------------#

def type2_group(x):
    one_word = ''
    two_word = ''
    
    one_word = list(x)[-1]
    if len(str(x)) > 1:
        two_word = list(x)[-2] + list(x)[-1]
    
    if x in ['招商','建行','北银','浦发','民生','华夏'] or two_word in ['银行']:
        return '银行'
    elif x in ['德邦','圆通','邮政','中通','韵达','宅急送','全峰','其它快递','申通','园通']:
        return '快递'
    elif x in ['好贷网','玖富','花呗','恒昌','捷信','宜信','借贷宝','分期乐','借贷保','马上','信用宝','融360','白条','贷小秘','现金巴士','佰仟'] \
                or one_word in ['贷'] or two_word in ['分期','贷款']:
        return '贷款相关'
    elif x in ['麦当劳','其它外卖','美团','肯德基','饿了么','必胜','广发']:
        return '外卖'
    elif x in ['信通','新联','通信']:
        return '通信'
    elif x in ['信和','汉力','地产','融信']:
        return '地产'
    elif x in ['信用卡','百世汇通']:
        return '信用卡'
    elif x in ['普通标记']:
        return '普通标记'
    elif x in ['骚扰','诈骗','套现','养卡','抵押','黑名单','法院','黑户','其它催收','律师']:
        return '不良标识'
    elif x in ['广告推销']:
        return '广告推销'
    else:
        return '其他'

df['type2_1'] = df.apply(lambda x: type2_group(x['type2']), axis = 1)


### 保存清洗完的数据
# 筛选符合时间范围
drop_var = ['start_time_fee','comm_plac_fee']
df = df.drop(drop_var, axis = 1)
df = df[df['app_date'] >= df['start_time']]
df = df.drop_duplicates()
df.to_csv('df_clean.csv', encoding = 'gbk', index = False)



