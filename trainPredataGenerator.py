# -*- coding: utf-8 -*-
"""
Created on Thu Dec 14 21:23:00 2017

@author: 56390
"""
import os;
import pandas as pd;
import numpy as np;
from pandas.tseries.offsets import Second;
#from lxyTools.preprocess import rawDataProcess;
#import lxyTools;
#from imp import reload;
#def naProcess(train,pre):
#    dataset=train.append(pre).copy();
#    temp=dataset.count();
#    nalist=temp[temp<dataset.shape[0]].index.tolist();
#    
#    print('数据列数:　',dataset.shape[1]);
#    print('含有空值列数:　',len(nalist));
#    
#    for col in nalist:
#        dataset[col+'_na']=0;
#        dataset.loc[dataset[col].isnull(),col+'_na']=1;
#        dataset.loc[dataset[col].isnull(),col]=dataset[col].mean();
#    return [dataset.iloc[0:train.shape[0]],dataset.iloc[train.shape[0]:dataset.shape[0]]];
#
#
#
#

def timeAdd(x,offset):
    result=pd.to_datetime(x,format='%Y%m%d%H%M%S')+offset*Second();
    result=str(result).replace('-','').replace(' ','').replace(':','');
    return int(result);
def timeDif(x,y):
    changedFlag=False;    
    if type(x)==pd._libs.tslib.Timestamp:
         changedFlag=True;
    elif type(x)==pd.Series:
        if x.dtype=='<M8[ns]':
            changedFlag=True;
    timex=pd.Series(x);
    timey=pd.Series(y);
    if not changedFlag:
        timex=toTime(timex);
        timey=toTime(timey);
        
    result=timex-timey;

    difday=result.apply(lambda x:x.days);
    difsecond=result.apply(lambda x:x.seconds);

    return difday*3600*24+difsecond;

def toTime(x):
    xcp=pd.Series(x);
    maxVal=xcp.max();
    if len(str(maxVal))==14:
        return pd.to_datetime(xcp.apply(lambda x:str(x)),format='%Y%m%d%H%M%S');
#        except:
#            print('The wrong value is:',x)
    elif len(str(maxVal))==16:
        return pd.to_datetime(xcp.apply(lambda x:str(x)[0:14]),format='%Y%m%d%H%M%S')
    elif len(str(maxVal))==8:
        return pd.to_datetime(xcp.apply(lambda x:str(x)[0:14]),format='%Y%m%d')
    else:
        print('err format');
        return -1;

def timeDifEnumerate(dataset,collist):
    result=dataset.copy()
    datasetcp=dataset[collist].copy();

    columns=datasetcp.columns;
    datasetcp=datasetcp.astype(np.int64);
    for i in range(0,datasetcp.shape[1]): 
        print(i,'/',datasetcp.shape[1]);
        for j in range(i+1,datasetcp.shape[1]):
            result[columns[i]+'-'+columns[j]]=timeDif(datasetcp.iloc[:,i],datasetcp.iloc[:,j]);
    return result;


def colDifEnumerate(dataset,collist):    
    datasetcp=dataset.copy();    
    meanall=datasetcp[collist].mean();
    count1=0;
    count2=0;
    for i in range(0,len(collist)-1):
        mean1=meanall[i];
        mean2=meanall[i+1];
        if abs(mean1-mean2)<0.5*min(abs(mean1),abs(mean2)):
            count1+=1;
            datasetcp.insert(loc=datasetcp.shape[1],column=collist[i]+'-'+collist[i+1]+'_colDif',value=datasetcp[collist[i]]-datasetcp[collist[i+1]]); 
#            datasetcp.insert(loc=datasetcp.shape[1],column=collist[i]+'*'+collist[i+1]+'_colDif',value=datasetcp[collist[i]]*datasetcp[collist[i+1]]); 
#            datasetcp.insert(loc=datasetcp.shape[1],column=collist[i]+'+'+collist[i+1]+'_colDif',value=datasetcp[collist[i]]-datasetcp[collist[i+1]]); 
#            
        if abs(mean1+mean2)<0.5*min(abs(mean1),abs(mean2)):
            count2+=1;
            datasetcp.insert(loc=datasetcp.shape[1],column=collist[i]+'+'+collist[i+1]+'_colDif',value=datasetcp[collist[i]]+datasetcp[collist[i+1]]); 
    print('通过两两做差产生的新的列数量: ',count1);
    print('通过两两相加产生的新的列数量: ',count2);
    return datasetcp;



def timeSeriesProcess(dataset):
    timelist=[];
    for col in dataset.columns:
        if dataset[col].dtype!='O':
            maxVal=dataset[col].max();
            if not maxVal is np.nan: 
                if '2017' == str(int(maxVal))[0:4]: 
                    timelist.append(col);

    print('len of timeSeries: ',len(timelist));
    datasetcp=dataset.copy();
    datasetcp.loc[datasetcp['210X24']==20166616661666,'210X24']=datasetcp.loc[datasetcp['210X24']==20166616661666,'210X204'].apply(lambda x:timeAdd(x,-438));
    

    temp=datasetcp[timelist].dropna().astype(np.int64);
    datasetcp.loc[datasetcp['210X213'].isnull(),'210X213']=datasetcp[datasetcp['210X213'].isnull()]['210X204'];
    meanVal=timeDif(temp['210X215'],temp['210X205']).mean();
    datasetcp.loc[datasetcp['210X215'].isnull(),'210X215']=datasetcp[datasetcp['210X215'].isnull()]['210X205'].apply(lambda x:timeAdd(x,meanVal));

    for col in timelist:
        if datasetcp[col].count()<datasetcp.shape[0]:
            datasetcp.loc[datasetcp[col].isnull(),col]=temp[col][0:temp.shape[0]-temp.shape[0]%2-1].median();


    datasetcp=timeDifEnumerate(datasetcp,timelist);

    for col in timelist:
        #提取月份，日期
        datasetcp.insert(loc=datasetcp.shape[1],column=col+'_month',value=datasetcp[col].apply(lambda x:str(x)[4:6]).astype(int));
        datasetcp.insert(loc=datasetcp.shape[1],column=col+'_day',value=datasetcp[col].apply(lambda x:str(x)[6:8]).astype(int));

        #提取当前小时，以及现在是白天还是晚上
        if len(str(int(datasetcp[col].max())))>=10:
            datasetcp.insert(loc=datasetcp.shape[1],column=col+'_hour',value=datasetcp[col].apply(lambda x:str(x)[8:10]).astype(int));
            datasetcp.insert(loc=datasetcp.shape[1],column=col+'_nightorday',value=0);
            datasetcp.loc[datasetcp[col+'_hour']<9,col+'_nightorday']=1;
            datasetcp.loc[datasetcp[col+'_hour']>16,col+'_nightorday']=1;
        #提取今天是星期几，以及判断这一天是工作日还是休息日
        if len(str(int(datasetcp[col].max())))>=14:
            datasetcp.insert(loc=datasetcp.shape[1],column=col+'_dayofweek',value=pd.to_datetime(datasetcp[col].astype(np.int64).apply(lambda x:str(x)[0:14]),format='%Y%m%d%H%M%S').apply(lambda x:x.dayofweek));
        elif len(str(int(datasetcp[col].max())))==8:
            datasetcp.insert(loc=datasetcp.shape[1],column=col+'_dayofweek',value=pd.to_datetime(datasetcp[col].astype(np.int64).apply(lambda x:str(x)),format='%Y%m%d').apply(lambda x:x.dayofweek));
        else:
            datasetcp.insert(loc=datasetcp.shape[1],column=col+'_dayofweek',value=pd.to_datetime(datasetcp[col].astype(np.int64).apply(lambda x:str(x)),format='%Y%m%d%H%M%S').apply(lambda x:x.dayofweek));

        datasetcp.insert(loc=datasetcp.shape[1],column=col+'_isoffday',value=0);
        datasetcp.loc[datasetcp[col+'_dayofweek']>4,col+'_isoffday']=1;
    return datasetcp;  
'''    
    dropList=[];
    for col in datasetcp.columns:
        if col not in timelist and \
            '_month' not in col and \
            '_day' not in col and \
            '_hour' not in col and\
            '_dayofweek' not in col and\
            '_nightorday' not in col and\
            '_isoffday' not in col:
                dropList.append(col);
    datasetcp=datasetcp.drop(dropList,axis=1);
'''
      

    
    
