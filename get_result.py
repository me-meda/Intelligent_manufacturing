import pandas as pd
import numpy as np
from pandas import Series,DataFrame
from DataProcess import data_process
import matplotlib
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV
matplotlib.use('Agg')


'''读取数据'''
trainData = pd.DataFrame(pd.read_excel('D:/Tianchi/data/train.xlsx'))
testData = pd.DataFrame(pd.read_excel('D:/Tianchi/data/test.xlsx'))
# print(trainData.shape)
# print('-'*30)
# print(trainData.count())
trainData_x = pd.DataFrame(trainData.drop(['Y','ID'],axis =1))
testData_x = pd.DataFrame(testData.drop(['ID'],axis = 1))
y_true = trainData['Y']

'''数据处理'''

'''获取类属性'''
catList = []
for i in trainData_x.columns:
    if trainData_x[i].dtypes == object:
        catList.append(i)
# print(catList)
'''空值处理'''
# print(scalList)

col_missing = trainData_x.isnull().any()[trainData_x.isnull().any()].index
col_missing_test = testData_x.isnull().any()[testData_x.isnull().any()].index
print(col_missing)
print('-'*30)
print(col_missing_test)
cat_missing = [x for x in col_missing if x in catList]
cat_missing_test = [x for x in col_missing_test if x in catList]
# print('训练集中类型变量中含空值的列有：',cat_missing)
# print('测试集中类变量包含空值的有：',cat_missing_test)
'''为训练集和测试集增加表示是否为空的属性列'''
for i in col_missing:
    trainData_x[str(i) + '_isnull'] = 0
    trainData_x.loc[trainData_x.loc[:,i].isnull(),str(i)+'_isnull'] = 1
    trainData_x.loc[trainData_x.loc[:, i].isnull(), i] = trainData_x.loc[:, i].mean()
for i in col_missing:
    testData_x[str(i) + '_isnull'] = 0
    testData_x.loc[testData_x.loc[:, i].isnull(), str(i) + '_isnull'] = 1
for i in col_missing_test:
    testData_x.loc[testData_x.loc[:, i].isnull(), i] = testData_x.loc[:, i].mean()
# print(trainData_x.head(5))
'''判断空值是否都处理完'''
for i in testData_x.columns:
    if testData_x.loc[:,i].isnull() is True:
        print('There still have null in ',str(i))
# print(type(testData_x))
'''删去元素值完全相同的列'''
print('orginal_length_train:',trainData_x.shape[1])
'''处理训练数据'''
dropList = []
for i in trainData_x.columns:
    if len(trainData_x[i].unique()) == 1:
        dropList.append(i)
        trainData_x = trainData_x.drop([i],axis = 1)
print('Uliti_length_train:',trainData_x.shape[1])
'''处理测试数据'''
print('-'*30)
print('orginal_length_test:',testData_x.shape[1])

'''获取数值属性进行尺度调整'''
testData_x = testData_x.drop(dropList,axis = 1)
train_scal_x = pd.DataFrame(trainData_x.drop(catList,axis = 1))
scalList = train_scal_x.columns.tolist()
testData_x = pd.DataFrame(testData_x)
trainData_x = pd.DataFrame(trainData_x)
print('Uliti_length_test:',testData_x.shape[1])
'''对类型变量做统一转换'''
for i in catList:
    CatList = trainData_x[i].unique()
    for k in range(0,len(CatList)):
       for j in CatList:
            if j == CatList[k]:
                trainData_x.loc[trainData_x.loc[:,i] == j,i] = k
for i in catList:
    CatList = testData_x[i].unique()
    for k in range(0,len(CatList)):
       for j in CatList:
            if j == CatList[k]:
                testData_x.loc[testData_x.loc[:,i] == j,i] = k

trainData_x = pd.DataFrame(trainData_x)
testData_x = pd.DataFrame(testData_x)

# '''检测数据中为字符型的数据列后完成数据类型转换'''
# for i in trainData_x.columns:
#     if trainData_x[i].dtypes == str:
#         trainData_x[i] = trainData_x[i].astype(float)
# for i in testData_x.columns:
#     if testData_x[i].dtypes == str:
#         testData_x[i] = testData_x[i].astype(float)
# for i in testData_x.columns:
#      if testData_x[i].dtypes == str:
#          print('There still have str column in:' + str(i) +'\n')
'''onehot及尺度调整'''
Data_proccess = data_process(trainData_x,testData_x,catList,scalList)
TrainData_new = pd.DataFrame(Data_proccess.toTrainData())
TestData_new = pd.DataFrame(Data_proccess.toTestData())

TrainData_new.to_csv('D:/Tianchi/Temp_Data/TrainData_revised.csv',index = False)
TestData_new.to_csv('D:/Tianchi/Temp_Data/TestData_revised.csv',index = False)
print(TrainData_new.head(5))
print('-'*30)
print(TestData_new.head(5))
print('-'*30)
'''GridSearch调参'''
# XGB_Model = xgb.XGBRegressor()
# paramGrid = {'max_depth':[5],
#          'learning_rate':[0.05],
#          'n_estimators':[200],
#          'gamma':[0],
#          'min_child_weight':[1],
#          'max_delta_step':[0],
#          'subsample':[1],
#          'colsample_bytree':[1],
#          'colsample_bylevel':[1],
#          'reg_alpha':[0],
#          'reg_lambda':[1],
#          'scale_pos_weight':[1],
#          'base_score':[0.5],
#          }
# XGB_Model = GridSearchCV(XGB_Model,paramGrid,scoring='neg_mean_squared_error',cv =KFold(n_splits=10,random_state=1))
# XGB_Model.fit(TrainData_new,y_true)
# print(XGB_Model.best_params_)
'''模型训练与测试  KFold交叉验证'''
n_splits = 50
kf =KFold(n_splits, random_state=1)
ave_score = 0
for kTrainIndex, kTestIndex in kf.split(TrainData_new , y_true):
    kTrain_x = TrainData_new.iloc[kTrainIndex]
    kTrain_y = y_true.iloc[kTrainIndex]

    kTest_x = TrainData_new.iloc[kTestIndex]
    kTest_y = y_true.iloc[kTestIndex]

    clf = xgb.XGBRegressor(max_depth = 5,learning_rate = 0.05,n_estimators = 200)

    xgb_model = clf.fit(kTrain_x, kTrain_y)  # Training model
    y_pred = xgb_model.predict(kTest_x)
    each_score = mean_squared_error(kTest_y,y_pred)
    ave_score = ave_score + each_score/n_splits
    print('each model gets score is:',each_score)
print('The ave result is:',ave_score)

# '''结果提交'''
# clf = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100)
# xgb_model = clf.fit(TrainData_new, y_true)
#
# Y_Pred = xgb_model.predict(TestData_new)
# ans = pd.DataFrame({'ID':testData['ID'],
#                     'Y':Y_Pred})
# ans.to_csv('D:/Tianchi/result/ans.csv',index = False)
