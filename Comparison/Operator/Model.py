
import pandas as pd
import numpy as np
import scorecardpy as sc
from sklearn.model_selection import train_test_split
from hyperopt import hp
from hyperopt import hp, fmin, tpe, Trials
from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier


### 读取数据
data = pd.read_csv('yys_lstm_data.csv', encoding = 'gbk')
data.shape

## 数据处理1: 删除验证集数据
data1 = data[data['is_test'] == 0]
data1 = data1.drop(['is_test','cell_phone'], axis = 1)
data1 = data1.fillna(0)
data1.label.value_counts()

### 分训练测试集
data1_x = data1.drop('label', axis = 1)
data1_y = data1.label.copy()

X_train, X_test, y_train, y_test = train_test_split(data1_x, data1_y, test_size=0.3, stratify = data1_y, random_state=617)

### 变量筛选
train_dt = pd.concat([X_train, y_train], axis = 1)

data_ft = sc.var_filter(train_dt, y = 'label')
var_ft = data_ft.columns.tolist()[:-1]
print(len(var_ft))

######################################
###  xgboost 自动调参优化 hyperopt  ###
######################################

from sklearn.model_selection import cross_val_score
from xgboost.sklearn import XGBClassifier

def objective(params):
  
    xgb_model = XGBClassifier(objective = 'binary:logistic',
                              max_depth = int(params['max_depth']),
                              min_child_weight = int(params['min_child_weight']),
                              n_estimators = int(params['n_estimators']),
                              learning_rate = round(params['learning_rate'],3),
                              subsample = round(params['subsample'],3), 
                              colsample_bytree = round(params['colsample_bytree'],3), 
                              reg_alpha = round(params['reg_alpha'],3),
                              reg_lambda = round(params['reg_lambda'],3),
                              random_state = int(params['random_state']))
    
    # Perform n_folds cross validation
    metric = cross_val_score(xgb_model, X_train[var_ft], y_train, cv=5, scoring="roc_auc").mean()
    
    return -metric

### 給定參數範圍
space = {
    'learning_rate': hp.uniform('learning_rate', 0.01, 0.5),
    'colsample_bytree': hp.uniform('colsample_bytree', 0.5, 1),
    'subsample': hp.uniform('subsample', 0.5, 1),
    'min_child_weight': hp.randint('min_child_weight', 30, 200),
    'reg_lambda': hp.uniform('reg_lambda', 0.0, 100.0),
    'max_depth': hp.randint('max_depth', 3, 6),
    'n_estimators': hp.randint('n_estimators', 50, 500),
    'reg_alpha': hp.uniform('reg_alpha', 0.0, 100.0),
    'random_state': hp.randint('random_state', 1, 1000),
    'eval_metric':'auc'
}

MAX_EVALS = 100
XGB_best = fmin(fn = objective, space = space, algo = tpe.suggest, max_evals = MAX_EVALS, trials = Trials())

# 带入模型进行预测
model_xgb_hp = XGBClassifier(**XGB_best, n_jobs = -1)
model_xgb_hp.fit(X_train[var_ft], y_train)

# 畫 KS 和 AUC圖形
train_pred_xgb = model_xgb_hp.predict_proba(X_train[var_ft])[:,1]
sc.perf_eva(y_train, train_pred_xgb, title = 'train')

test_pred_xgb = model_xgb_hp.predict_proba(X_test[var_ft])[:,1]
sc.perf_eva(y_test, test_pred_xgb, title = 'test')

### 保存模型
import pickle
with open('model/model_xgb_best.pkl','wb') as f:
    pickle.dump(model_xgb_hp, f)


########################################
###  Lgbmboost 自动调参优化 hyperopt  ###
########################################


def lgbm_hp_fn(params):
    
    lgbm = LGBMClassifier(boosting_type = 'gbdt', 
                          objective = 'binary', 
                          n_estimators = int(params['n_estimators']),
                          learning_rate = round(params['learning_rate'],3),
                          max_depth = int(params['max_depth']),
                          num_leaves = int(params['num_leaves']),
                          subsample = round(params['subsample'],3), 
                          colsample_bytree = round(params['colsample_bytree'],3), 
                          reg_alpha = round(params['reg_alpha'],3),
                          reg_lambda = round(params['reg_lambda'],3),
                          random_state = int(params['random_state']))
    
    metric = cross_val_score(lgbm, X_train[var_ft], y_train, cv = 5, scoring = 'roc_auc').mean()
    #print(metric)
    
    return -metric

### 給定參數範圍
space = {
    'n_estimators':hp.randint("n_estimators", 10, 1000),
    'learning_rate':hp.uniform("learning_rate",0.01,0.5),
    'max_depth':hp.randint("max_depth", 3, 6),
    'num_leaves':hp.randint("num_leaves", 2, 64),
    'subsample':hp.uniform("subsample",0.5,1),
    'colsample_bytree':hp.uniform("colsample_bytree",0.5,1),
    'reg_alpha':hp.uniform("reg_alpha",0,10),
    'reg_lambda':hp.uniform("reg_lambda",0,10),
    'random_state':hp.randint("random_state",0,1000)
}

#进行贝叶斯调参
trials = Trials()
max_evals = 100
LGBM_best = fmin(lgbm_hp_fn, space, algo = tpe.suggest, max_evals = max_evals, trials = trials)

