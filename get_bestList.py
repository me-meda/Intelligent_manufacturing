# -*- coding: utf-8 -*-
"""
Created on Tue Jan  9 16:35:41 2018

@author: Xu Bing
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
from sklearn.metrics import mean_squared_error
def get_bestList(origList,TrainData,y_true):
    best_i = 80
    best_mse = 0.03
    for i in range(80,90):
        testList = origList[0:i]
        TrainDatacp = TrainData.copy()
        testData = TrainDatacp[testList]
        n_splits = 20
        kf =KFold(n_splits, random_state=1)
        ave_score_xgb = 0
        for kTrainIndex, kTestIndex in kf.split(testData , y_true):
            kTrain_x = testData.iloc[kTrainIndex]
            kTrain_y = y_true.iloc[kTrainIndex]
    
            kTest_x = testData.iloc[kTestIndex]
            kTest_y = y_true.iloc[kTestIndex]
    
            clf = xgb.XGBRegressor(max_depth = 3,learning_rate = 0.05,n_estimators = 500,subsample = 0.8)
        
            knn_model = clf.fit(kTrain_x, kTrain_y)
        
            y_pred_knn = knn_model.predict(kTest_x)
        
            each_score_knn = mean_squared_error(kTest_y, y_pred_knn)
        
            ave_score_xgb = ave_score_xgb + each_score_knn / n_splits
                              
        print('This mean xgb_mse is:',ave_score_xgb)
        if ave_score_xgb < best_mse:
            best_i = i
            best_mse = ave_score_xgb
    print('The best interval is:','[0:'+str(best_i)+']')
    print('The best mse is:',best_mse)