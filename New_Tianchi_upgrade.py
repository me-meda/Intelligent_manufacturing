# -*- coding: utf-8 -*-
"""
Created on Wed Jan  3 14:20:46 2018

@author: Xu Bing
"""

import pandas as pd
import numpy as np
from pandas import Series,DataFrame
from DataProcess import data_process
import matplotlib
import xgboost as xgb
from minepy import MINE


from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from TCDataProcess import TCdata_process
from New_DataProcess import rawDataProcess
from trainPredataGenerator import timeSeriesProcess
from trainPredataGenerator import colDifEnumerate
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFECV
import lightgbm as lgb
from get_bestList import get_bestList



from sklearn.feature_selection import mutual_info_regression
import copy
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KNeighborsRegressor
import seaborn as sns







'''读取数据'''
trainData = pd.read_excel('D:/Tianchi/data/train.xlsx')
testData = pd.read_excel('D:/Tianchi/data/test.xlsx')
testData_B = pd.read_excel('D:/Tianchi/data/test_B.xlsx')
# print(trainData.shape)
# print('-'*30)
# print(trainData.count())
trainData_x = trainData.drop(['Y','ID'],axis =1)
testData_x = testData.drop(['ID'],axis = 1)
y_true = trainData['Y']



#for i in trainData_x.columns:
#    if 100 < trainData_x[i].count() < 300:
#        print('The col contain nan is:',i)
#        print('The true num is:',trainData_x[i].count())
#        print('-'*30)




'''删除与测试集差异过大的列'''
#dropList = []
#for i in trainData_x.columns:
#    if trainData_x[i].dtypes == 'object':
#        dropList.append(i)
#    if trainData_x[i].max() == trainData_x[i].min():
#        dropList.append(i)
#    if trainData_x[i].count() == 0:
#        dropList.append(i)
#scaList = trainData_x.drop(dropList,axis = 1).columns.tolist()
#differList = []
#for col in scaList:
#    if abs(trainData[col].std() - testData[col].std()) > 5 or abs(trainData[col].mean() - testData[col].mean()) > 10:
#        differList.append(col)
#print('The length of differList is:',len(differList))
#trainData_x = trainData_x.drop(differList,axis = 1)
#testData_x = testData_x.drop(differList,axis = 1)






'''数据处理'''


enumerateData = TCdata_process(trainData_x,testData_x)
'''产生时间差，增添时间信息'''
enumerateData.dataSets = timeSeriesProcess(enumerateData.dataSets)
enumerateData.fillNan()
enumerateData.Drop_features()


'''产生数值属性之差的列'''
colDifList=[];
for col in trainData_x.columns:
    if trainData_x[col].dtype!='O' and col in enumerateData.dataSets.columns:
        colDifList.append(col);

enumerateData.dataSets=colDifEnumerate(enumerateData.dataSets,colDifList);







'''产生新的列记录样本中存在空值的数量'''
enumerateData.dataSets['na_sum']=0
for col in enumerateData.dataSets.columns:
    if '_na' in col:
        enumerateData.dataSets['na_sum']= enumerateData.dataSets['na_sum']+enumerateData.dataSets[col]
print(enumerateData.dataSets.shape)

'''原始筛选catList方法'''
# temp=enumerateData.dataSets.dtypes
# catlist=temp[temp=='object'].index.tolist()
'''通过列名筛选'''
catlist = []
for i in trainData_x.columns:
    if 'Tool' in i:
        catlist.append(i)
    elif 'TOOL' in i:
        catlist.append(i)
    elif 'tool' in i:
        catlist.append(i)

for col in enumerateData.dataSets.columns:
    if '_month' in col or '_day' in col or '_hour' in col or '_dayofweek' in col:
        catlist.append(col)
print('The length of catList is:',len(catlist))


sclist=enumerateData.dataSets.drop(catlist,axis=1).columns.tolist()

enumerateData.Datafit(catlist,sclist)

train_newData = enumerateData.toTrainData()
test_newData = enumerateData.toTestData()

#
# train_newData.to_csv('D:/Tianchi/TempData/train_newData.csv',index = False)
# test_newData.to_csv('D:/Tianchi/TempData/test_newData.csv',index = False)
#
#
#
print(train_newData.get_dtype_counts())
print('-'*30)
print(test_newData.get_dtype_counts())





'''根据mutual_info_regression筛选特征'''
#scoreList = pd.Series(mutual_info_regression(train_newData,y_true))
#scoreList.index = train_newData.columns
#RankingList = scoreList.sort_values(ascending=False)
##print('The original length of the rankingList is:',len(RankingList))
##for i in RankingList.index:
##    if RankingList[i] == 0:
##        RankingList = RankingList.drop(i)
##print('The revised length of the rankingList is:',len(RankingList))
#train_newData1 = train_newData[RankingList[0:5000].index]
#test_newData1 = test_newData[RankingList[0:5000].index]



'''根据最大信息系数筛选特征'''
maxInfo = MINE()
scoreList = []
for i in train_newData.columns:
    maxInfo.compute_score(train_newData[i],y_true)
    scoreList.append(maxInfo.mic())
scoreList = pd.Series(scoreList)
scoreList.index = train_newData.columns
RankingList = scoreList.sort_values(ascending=False)
print('The original length of the rankingList is:',len(RankingList))
#for i in RankingList.index:
#    if RankingList[i] == 0:
#        RankingList = RankingList.drop(i)
#print('The revised length of the rankingList is:',len(RankingList))
train_newData1 = train_newData[RankingList.index]
test_newData1 = test_newData[RankingList.index]

print(train_newData1.head(5))
print('-'*30)
print(test_newData1.head(5))

'''删除两列相关性过大的列，仅保留一列'''
colList = train_newData1[RankingList[0:5000].index].columns
print('The original length is:',len(colList))
corrList = []
cormat = train_newData1.corr()
for col in colList:
    if col not in corrList:
        temp = cormat[col]
        tempList = temp[abs(temp) > 0.9].index.tolist()
        tempList.remove(col)    #不删除自身
        for i in tempList:
            if i not in corrList:
                corrList.append(i)
print('The num of high corr list is:',len(corrList))

train_newData2 = train_newData1.drop(corrList,axis = 1)
test_newData2 = test_newData1.drop(corrList,axis = 1)

print(train_newData2.shape)
print(test_newData2.shape)




'''对lightGBM调参'''
#LGB_Model = lgb.LGBMRegressor()
#paramGrid = {'boosting_type':['gbdt'],
#          'num_leaves':[31],
#          'max_depth':[8],
#          'learning_rate':[0.05],
#          'n_estimators':[200],
#          'random_state':[666]
#                   }
#LGB_Model = GridSearchCV(LGB_Model,paramGrid,scoring='neg_mean_squared_error',cv = KFold(n_splits=10,random_state=666))
#LGB_Model.fit(train_newData1,y_true)
#print(LGB_Model.best_params_)



'''根据模型给出的重要性筛选特征'''
xgb_model = xgb.XGBRegressor(max_depth = 3,learning_rate = 0.05,n_estimators = 500,subsample = 0.8)
xgb_model.fit(train_newData2,y_true)
xgb_feaimp = pd.Series(xgb_model.feature_importances_)
xgb_feaimp.index = train_newData2.columns
selectedList = xgb_feaimp.sort_values(ascending=False)
selectedList = selectedList.index


print(selectedList)

'''由第二个模型筛选特征'''
#lgb_model = lgb.LGBMRegressor(n_estimators = 200,max_depth = 8 ,random_state = 666 ,boosting_type = 'gbdt',learning_rate = 0.05 ,num_leaves = 31 )
#lgb_model.fit(train_newData2,y_true)
#lgb_feaimp = pd.Series(lgb_model.feature_importances_)
#lgb_feaimp.index = train_newData2.columns
#chooseList = lgb_feaimp.sort_values(ascending = False)
#
#
#
#finalList = []
#for col in selectedList.index:
#    if col in chooseList.index:
#        finalList.append(col)
#
#print('The length of the finalList is:',len(finalList))



#finalList = finalList[0:30]
#wantedList = []
#for i in finalList:
#    if (train_newData1[i].max() - test_newData1[i].max()) >0:
#        if (test_newData1[i].max() - train_newData1[i].min())/(test_newData1[i].max() - test_newData1[i].min()) > 0.8:
#            wantedList.append(i)
#    elif (train_newData1[i].max() - test_newData1[i].max())  < 0:
#        if (train_newData1[i].max() - test_newData1[i].min())/(test_newData1[i].max() - test_newData1[i].min()) > 0.8:
#            wantedList.append(i) 
#print('The length of the wantedList is:',len(wantedList))
#print('The wantedList is:',wantedList)



New_trainData = train_newData1[selectedList[0:81]]
New_testData = test_newData1[selectedList[0:81]]


'''GridSearch调参'''
XGB_Model = xgb.XGBRegressor()
paramGrid = {'max_depth':[3],
          'learning_rate':[0.1],
          'n_estimators':[200,500,800],
          'gamma':[0],
          'min_child_weight':[1],
          'max_delta_step':[0],
          'subsample':[0.8],
          'colsample_bytree':[1],
          'colsample_bylevel':[1],
          'reg_alpha':[0],
          'reg_lambda':[1],
          'scale_pos_weight':[1],
          'base_score':[0.5],
          }
XGB_Model = GridSearchCV(XGB_Model,paramGrid,scoring='neg_mean_squared_error',cv = KFold(n_splits=10,random_state=1))
XGB_Model.fit(New_trainData,y_true)
print(XGB_Model.best_params_)


get_bestList(selectedList,train_newData1,y_true)





'''模型训练及本地cv'''
n_splits = 20
kf =KFold(n_splits, random_state=1)
ave_score_xgb = 0
ave_score_knn = 0


for kTrainIndex, kTestIndex in kf.split(New_trainData , y_true):
    kTrain_x = New_trainData.iloc[kTrainIndex]
    kTrain_y = y_true.iloc[kTrainIndex]

    kTest_x = New_trainData.iloc[kTestIndex]
    kTest_y = y_true.iloc[kTestIndex]

    clf = xgb.XGBRegressor(max_depth = 3,learning_rate = 0.05,n_estimators = 500,subsample = 0.8)
#    clf2 = lgb.LGBMRegressor(max_depth = 5,learning_rate = 0.05,n_estimators = 200)
#    clf2 =KNeighborsRegressor(n_neighbors=10, algorithm='ball_tree')

    #xgb_model = clf.fit(kTrain_x, kTrain_y)  # Training model
    # lgb_model = clf2.fit(kTrain_x, kTrain_y)
    knn_model = clf.fit(kTrain_x, kTrain_y)


    #y_pred_xgb = xgb_model.predict(kTest_x)
    #y_pred_lgb = lgb_model.predict(kTest_x)
    y_pred_knn = knn_model.predict(kTest_x)

    # each_score_xgb = mean_squared_error(kTest_y,y_pred_xgb)
    # each_score_lgb = mean_squared_error(kTest_y,y_pred_xgb)
    each_score_knn = mean_squared_error(kTest_y, y_pred_knn)

    # ave_score_xgb = ave_score_xgb + each_score_xgb/n_splits
    # ave_score_lgb = lgb_ave_score + each_score_lgb / n_splits
    ave_score_knn = ave_score_knn + each_score_knn / n_splits


    print('each xgb_model gets score is:',each_score_knn)
    print('-'*30)
    # print('each lgb_model gets score is:', each_score_lgb)
print('The mean xgb_mse is:',ave_score_knn)









print(New_trainData.shape)

print(New_testData.shape)

'''结果提交'''
clf = xgb.XGBRegressor(max_depth=3, learning_rate=0.1, n_estimators=100)
xgb_model = clf.fit(New_trainData, y_true)
Y_Pred = xgb_model.predict(New_testData)
ans = pd.DataFrame({'ID':testData['ID'],
                     'Y':Y_Pred})
ans.to_csv('D:/Tianchi/result/ans_0105.csv',index = False,header = False)




