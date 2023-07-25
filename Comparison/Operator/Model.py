
import pandas as pd
import numpy as np
import scorecardpy as sc
from sklearn.model_selection import train_test_split, cross_val_score
from hyperopt import hp
from hyperopt import hp, fmin, tpe, Trials
from sklearn.linear_model import LogisticRegression
from xgboost.sklearn import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier
from keras.layers import Input,Conv1D,MaxPooling1D,Dense,Flatten,concatenate,Dropout,GlobalMaxPooling1D,LSTM
from keras.models import Sequential,Model
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import roc_auc_score
import pickle


### 读取数据
data = pd.read_csv('yys_lstm_data.csv', encoding = 'gbk')
data.shape

### 数据处理1: 删除验证集数据
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

### 带入模型进行预测
model_xgb_hp = XGBClassifier(**XGB_best, n_jobs = -1)
model_xgb_hp.fit(X_train[var_ft], y_train)
train_pred_xgb = model_xgb_hp.predict_proba(X_train[var_ft])[:,1]
test_pred_xgb = model_xgb_hp.predict_proba(X_test[var_ft])[:,1]


### 畫 KS 和 AUC圖形
sc.perf_eva(y_train, train_pred_xgb, title = 'train')
sc.perf_eva(y_test, test_pred_xgb, title = 'test')

### 保存模型
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

### 进行贝叶斯调参
trials = Trials()
max_evals = 100
LGBM_best = fmin(lgbm_hp_fn, space, algo = tpe.suggest, max_evals = max_evals, trials = trials)

### 带入模型进行预测
model_lgbm_hp = LGBMClassifier(boosting_type='gbdt', 
                            objective='binary', 
                            n_estimators = int(LGBM_best['n_estimators']),
                            learning_rate = LGBM_best['learning_rate'],
                            max_depth = int(LGBM_best['max_depth']),
                            num_leaves = int(LGBM_best['num_leaves']),
                            subsample = LGBM_best['subsample'], 
                            colsample_bytree = LGBM_best['colsample_bytree'], 
                            reg_alpha = LGBM_best['reg_alpha'],
                            reg_lambda = LGBM_best['reg_lambda'],
                            random_state = int(LGBM_best['random_state'])
                            )
model_lgbm_hp.fit(X_train[var_ft], y_train)

### 带入模型进行预测
train_pred_lgbm = model_lgbm_hp.predict_proba(X_train[var_ft])[:,1]
test_pred_lgbm = model_lgbm_hp.predict_proba(X_test[var_ft])[:,1]


### 畫 KS 和 AUC圖形
sc.perf_eva(y_train, train_pred_lgbm)
sc.perf_eva(y_test, test_pred_lgbm)

### 保存模型
with open('model/model_lgbm_best.pkl','wb') as f:
    pickle.dump(model_lgbm_hp, f)



#########################################
###     GBDT 自动调参优化 hyperopt     ###
#########################################

def gbdt_hp_fn(params):
    
    gbdt = GradientBoostingClassifier(
                                      n_estimators = int(params['n_estimators']),
                                      learning_rate = round(params['learning_rate'],3),
                                      max_depth = int(params['max_depth']),
                                      subsample = round(params['subsample'],3), 
                                      min_samples_split = int(params['min_samples_split']),
                                      random_state = int(params['random_state'])
                                     )
    
    metric = cross_val_score(gbdt, X_train[var_ft], y_train, cv = 5, scoring = 'roc_auc').mean()
    #print(metric)
    
    return -metric

### 給定參數範圍
space = {
    'n_estimators':hp.randint("n_estimators", 10, 1000),
    'learning_rate':hp.uniform("learning_rate",0.01,0.5),
    'max_depth':hp.randint("max_depth", 3, 6),
    'subsample':hp.uniform("subsample",0.5,1),
    'min_samples_split':hp.randint("min_samples_split",1,1000),
    'random_state':hp.randint("random_state",0,1000)
}

### 进行贝叶斯调参
trials = Trials()
max_evals = 100
GBDT_best = fmin(gbdt_hp_fn, space, algo = tpe.suggest, max_evals = max_evals, trials = trials)

### 带入模型进行预测
model_gbdt_hp = GradientBoostingClassifier(
                                            n_estimators = int(GBDT_best['n_estimators']),
                                            learning_rate = GBDT_best['learning_rate'],
                                            max_depth = int(GBDT_best['max_depth']),
                                            subsample = GBDT_best['subsample'], 
                                            min_samples_split = int(GBDT_best['min_samples_split']), 
                                            random_state = int(GBDT_best['random_state'])
                                            )
model_gbdt_hp.fit(X_train[var_ft], y_train)
train_pred_gbdt = model_gbdt_hp.predict_proba(X_train[var_ft])[:,1]
test_pred_gbdt = model_gbdt_hp.predict_proba(X_test[var_ft])[:,1]


### 畫 KS 和 AUC圖形
sc.perf_eva(y_train, train_pred_gbdt)
sc.perf_eva(y_test, test_pred_gbdt)

### 保存模型
with open('model/model_gbdt_best.pkl','wb') as f:
    pickle.dump(model_gbdt_hp, f)



#########################################
###             Keras CNN             ###
#########################################

## 数据处理2: 归一化
X_train2 = pd.DataFrame()
X_test2 = pd.DataFrame()

for i in X_train.columns.tolist():
    max_value = max(X_train[i])
    X_train2[i] = X_train[i].apply(lambda x: x/max_value if max_value > 0 else 0).fillna(0)
    X_test2[i] = X_test[i].apply(lambda x: x/max_value if max_value > 0 else 0).fillna(0)

## 维度转换
x_train2 = np.reshape(X_train2.values,(X_train2.shape[0],21,42))
x_test2 = np.reshape(X_test2.values,(X_test2.shape[0],21,42))
y_train2 = np.array(y_train)
y_test2 = np.array(y_test)

### 定義CNN模型
def model_cnn2_rawdata(conv1d, batch_size_, dropout_):
    model_cnn2 = Sequential([

        ## 第一层
        Conv1D(filters = conv1d, kernel_size = 2, padding='same', activation='relu', input_shape = (21, 42)),
        MaxPooling1D(pool_size=2),
        Dropout(dropout_),
        
        ## 全链接
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(dropout_),
        Dense(1, activation='sigmoid')
        ]
    )
    model_cnn2.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
    model_cnn2.fit(x_train2, y_train2, epochs=100, batch_size=batch_size_, verbose = 0)
    
    return model_cnn2

### 利用網格搜索
result_cnn2 = pd.DataFrame()

for i in [9, 18, 36, 64, 72, 100, 128, 144]:
    for j in [200, 500, 750, 1000]:
        for k in [0.5, 0.6, 0.7, 0.8]:
        
            ## 训练模型
            model_cnn2 = model_cnn2_rawdata(i, j, k)

            ## 训练集
            score_train = model_cnn2.predict(x_train2)
            train_auc_cnn = roc_auc_score(y_train2, score_train)

            ## 测试集
            score_test = model_cnn2.predict(x_test2)
            test_auc_cnn = roc_auc_score(y_test2, score_test)
            
            if test_auc_cnn > 0.61:
                result_cnn2 = result_cnn2.append({'conv1d':i, 
                                                  'batch_size':j,
                                                  'dropout': k,
                                                  'train_auc':train_auc_cnn,
                                                  'test_auc':test_auc_cnn
                                                 }, ignore_index = True)
            if test_auc_cnn > 0.615:
                name = str(i) + '_' + str(j) + '_' + str(k)
                model_cnn2.save('./model/model_cnn_best_' + name)

### CNN 最优参数
# model_cnn_best = Sequential([

#     ## 第一层
#     Conv1D(filters = 9, kernel_size = 2, padding='same', activation='relu', input_shape = (21, 42)),
#     MaxPooling1D(pool_size=2),
#     Dropout(0.5),

#     ## 全链接
#     Flatten(),
#     Dense(128, activation='relu'),
#     Dropout(0.5),
#     Dense(1, activation='sigmoid')
#     ]
# )
# model_cnn_best.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])
# model_cnn_best.fit(x_train2, y_train2, epochs=100, batch_size=200, verbose = 0)

### 保存模型
#model_cnn_best.save('./model/model_cnn_best')

### 读取模型
model_cnn_best = keras.models.load_model("./model/model_cnn_best_9_200_0.5")

### 模型預測
train_pred_cnn = model_cnn_best.predict(x_train2)
train_auc_cnn = roc_auc_score(y_train2, train_pred_cnn)
print(train_auc_cnn)

test_pred_cnn = model_cnn_best.predict(x_test2)
test_auc_cnn = roc_auc_score(y_test2, test_pred_cnn)
print(test_auc_cnn)

#########################################
###             Keras LSTM            ###
#########################################

### 定义LSTM函数
def model_lstm_rawdata(LSTM_, batch_size_):
    model_lstm = Sequential()
    model_lstm.add(LSTM(LSTM_, batch_input_shape=(None, 21, 42), return_sequences=True))
    model_lstm.add(LSTM(LSTM_, activation="sigmoid"))
    model_lstm.add(Dense(1, activation="sigmoid"))
    model_lstm.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
    model_lstm.fit(x_train2, y_train2, epochs=100, batch_size=batch_size_, verbose = 0)
    
    return model_lstm

### 利用網格搜索
result_lstm = pd.DataFrame()

for i in [6, 12, 18, 24, 30, 36]:
    for j in [100, 200, 500, 750, 1000]:
        
        ## 训练模型
        model_lstm = model_lstm_rawdata(i, j)
        
        ## 训练集
        score_train = model_lstm.predict(x_train2)
        train_auc_lstm = roc_auc_score(y_train2, score_train)
        
        ## 测试集
        score_test = model_lstm.predict(x_test2)
        test_auc_lstm = roc_auc_score(y_test2, score_test)
        
        if test_auc_lstm > 0.61:
            result_lstm = result_lstm.append({'lstm_param':i, 
                                              'batch_size':j,
                                              'train_auc':train_auc_lstm,
                                              'test_auc':test_auc_lstm
                                             }, ignore_index = True)
            
            model_lstm.save('./model/model_lstm_best_' + str(i) + '_' + str(j))


# model_lstm_best = Sequential()
# model_lstm_best.add(LSTM(24, batch_input_shape=(None, 21, 42), return_sequences=True))
# model_lstm_best.add(LSTM(24, activation="sigmoid"))
# model_lstm_best.add(Dense(1, activation="sigmoid"))
# model_lstm_best.compile(loss='binary_crossentropy', optimizer='adam', metrics=['binary_accuracy'])
# model_lstm_best.fit(x_train2, y_train2, epochs=100, batch_size=200, verbose = 0)

### 保存模型
#model_lstm_best.save('./model/model_lstm_best')

### 读取模型
model_lstm_best = keras.models.load_model("./model/model_lstm_best_30_500")
from sklearn.metrics import roc_auc_score
train_pred_lstm = model_lstm_best.predict(x_train2)
train_auc_lstm = roc_auc_score(y_train2, train_pred_lstm)
print(train_auc_lstm)

test_pred_lstm = model_lstm_best.predict(x_test2)
test_auc_lstm = roc_auc_score(y_test2, test_pred_lstm)
print(test_auc_lstm)

#########################################
###        Logistic Regression        ###
#########################################

### 模型训练
model_lr = LogisticRegression().fit(X_train[var_ft], y_train)

### 模型预测
train_pred_lr = model_lr.predict_proba(X_train[var_ft])[:,1]
train_auc_lr = roc_auc_score(y_train, train_pred_lr)
print(train_auc_lr)

test_pred_lr = model_lr.predict_proba(X_test[var_ft])[:,1]
test_auc_lr = roc_auc_score(y_test, test_pred_lr)
print(test_auc_lr)

### 保存模型
with open('model/model_lr_best.pkl','wb') as f:
    pickle.dump(model_lr, f)


#########################################
###    模型兩兩組合預測值，在執行LR     ###
#########################################

### 合并所有预测结果
train_all_pred = pd.DataFrame()
train_all_pred['label'] = y_train.copy()
train_all_pred['lr'] = train_pred_lr.copy()
train_all_pred['xgb'] = train_pred_xgb.copy()
train_all_pred['lgbm'] = train_pred_lgbm.copy()
train_all_pred['gbdt'] = train_pred_gbdt.copy()
train_all_pred['cnn'] = train_pred_cnn.copy()
train_all_pred['lstm'] = train_pred_lstm.copy()
print(train_all_pred.shape)

# test
test_all_pred = pd.DataFrame()
test_all_pred['label'] = y_test.copy()
test_all_pred['lr'] = test_pred_lr.copy()
test_all_pred['xgb'] = test_pred_xgb.copy()
test_all_pred['lgbm'] = test_pred_lgbm.copy()
test_all_pred['gbdt'] = test_pred_gbdt.copy()
test_all_pred['cnn'] = test_pred_cnn.copy()
test_all_pred['lstm'] = test_pred_lstm.copy()
print(test_all_pred.shape)

### 两两模型预测值重新训练
model_name = ['lr','xgb','lgbm','gbdt','cnn','lstm']
tmp_model_name = model_name.copy()
result_mix_auc = pd.DataFrame()
cnt = 0

for i in model_name:
    tmp_name = []
    tmp_name.append(i)
    tmp_model_name.remove(i)
    
    for j in tmp_model_name:

        tmp_name.append(j)
        result_mix_auc.loc[cnt,'model_name'] = '_'.join(tmp_name)
        tmp_tr_data = train_all_pred[tmp_name]
        tmp_ts_data = test_all_pred[tmp_name]

        ### 模型训练
        model_mix_lr = LogisticRegression().fit(tmp_tr_data, train_all_pred['label'])

        ### 模型预测
        train_pred_lr = model_mix_lr.predict_proba(tmp_tr_data)[:,1]
        train_auc_lr = roc_auc_score(train_all_pred['label'], train_pred_lr)
        result_mix_auc.loc[cnt,'train_auc'] = train_auc_lr
        
        test_pred_lr = model_mix_lr.predict_proba(tmp_ts_data)[:,1]
        test_auc_lr = roc_auc_score(test_all_pred['label'], test_pred_lr)
        result_mix_auc.loc[cnt,'test_auc'] = test_auc_lr
        
        tmp_name.remove(j)
        
        cnt += 1
        
    cnt += 1

print(result_mix_auc)




