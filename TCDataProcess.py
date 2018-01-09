import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import OneHotEncoder
import numpy as np
class TCdata_process:
    def __init__(self,TrainData,TestData):
        assert type(TrainData) == pd.DataFrame
        assert type(TestData)  == pd.DataFrame

        self.TrainData = TrainData
        self.TestData =TestData

        self.dataSets = self.TrainData.copy().append(self.TestData.copy())   #此步很重要，组合训练集与测试集  方便统一进行空值及增删列的操作，统一训练集和测试集的列名



    def Drop_features(self):
        vividlen = self.dataSets.count()
        '''删除全空的列'''
        nullList = vividlen[vividlen == 0].index.tolist()
        # print('The null List are:',nullList)
        print('The null List num is:',len(nullList))
        self.dataSets = self.dataSets.drop(nullList,axis = 1)

        '''删除列中元素均相同的列'''
        AllsameList = []
        for i in self.dataSets.columns:
            if self.dataSets[i].max() == self.dataSets[i].min():
                AllsameList.append(i)
        
        self.dataSets = self.dataSets.drop(AllsameList,axis = 1)
        print('删除列中元素都相同的列后：',self.dataSets.shape)
        print('-'*30)
        '''删除训练集与测试集相差过大的列'''
        # dropList = []
        # for i in self.TrainData.columns:
        #     if self.TrainData[i].dtypes == 'object':
        #         dropList.append(i)
        #     if self.TrainData[i].max() == self.TrainData[i].min():
        #         dropList.append(i)
        #     if self.TrainData[i].count() == 0:
        #         dropList.append(i)
        # scaList = self.TrainData.drop(dropList, axis=1).columns.tolist()

        # differList = []
        # for col in scaList:
        #     if abs(self.TrainData[col].std() - self.TestData[col].std()) > 5 or abs(self.TrainData[col].mean() - self.TestData[col].mean()) > 10:
        #         differList.append(col)
        # print('initial differList:',len(differList))
        # for i in differList:
        #     if i not in self.dataSets.columns:
        #         differList.remove(i)
        # print('ultimate differList:',len(differList))
        #
        # self.dataSets = self.dataSets.drop(differList,axis = 1)
        # print('删除差异列后',self.dataSets)
        
        print('数据集的空值情况',self.dataSets.count())
        for i in self.dataSets.columns:
            if 100 < self.dataSets[i].count() < 300:
                print('The col contain nan is:',i)
                print('The true num is:',self.dataSets[i].count())
                print('-'*30)

        
        
        
        
        
    def fillNan(self):
        
        
        
        print('删除空值过多的列之前：',self.dataSets.shape)
        '''删除含空值过多的列'''
        lostList = []
        for i in self.dataSets.columns:
            if self.dataSets[i].count() < 300 :
                lostList.append(i)
        print('The lostList is',lostList)
        self.dataSets = self.dataSets.drop(lostList,axis = 1)
        print('删除含空值过多的列后：',self.dataSets.shape)
        
        
        
        temp = self.dataSets.count()
        naList = temp[temp < self.dataSets.shape[0]]
        naList = naList[naList>0].index.tolist()
        print('数据列数:　', self.dataSets.shape[1]);
        print('含有空值列数(非全空):　', len(naList));
        
        
        for i in naList:
            '''增添新的列记录空值的情况'''
            self.dataSets[i + '_na'] = 0
            self.dataSets.loc[self.dataSets[i].isnull(),i + '_na'] = 1
            if self.dataSets[i].dtypes != 'object':
                self.dataSets.loc[self.dataSets[i].isnull(),i] = self.dataSets[i].mean()
            else:
                self.dataSets.loc[self.dataSets[i].isnull(),i] = self.dataSets[i].mode()[0]

    def Datafit(self,CatList,ScaList):

        assert type(CatList) == list
        assert type(ScaList) == list

        self.CatList = CatList
        self.ScaList = ScaList

        self.enc = OneHotEncoder(dtype = np.int32)
        self.scaler = MinMaxScaler()
        '''对类型变量中的数据转换为整型数据'''
        temp = self.dataSets[CatList].copy()
        self.nameList = []
        for col in CatList:
            i = 0
            for val in self.dataSets[col].unique():
                i = i + 1
                temp.loc[self.dataSets[col] == val, col] = i
                self.nameList.append(col+'_cat_'+str(val))
        self.dataSets[CatList] = temp.values
        print('catList num is:',len(self.nameList))
        if len(CatList) >= 1:
            self.enc.fit(temp)
        if len(self.ScaList) >= 1:
            self.scaler.fit(self.dataSets[ScaList])

    def toTrainData(self):
        allList = self.CatList + self.ScaList
        trainData = self.dataSets.iloc[0:self.TrainData.shape[0]]
        print('TrainData size is:',trainData.shape)
        Train_Re = trainData.drop(allList,axis = 1)
        if len(self.CatList) >= 1:
            enctransData = pd.DataFrame(self.enc.transform(trainData[self.CatList]).toarray())
            Train_Re[self.nameList] = enctransData
             
        if len(self.ScaList) >= 1:
            scatransData = pd.DataFrame(self.scaler.transform(trainData[self.ScaList]))
            Train_Re[self.ScaList] = scatransData
        print('Train_re shape: ', Train_Re.shape)
        return Train_Re

    def toTestData(self):
        allList = self.CatList + self.ScaList
        testData = self.dataSets.iloc[self.TrainData.shape[0]:]
        print('TestData size is:',testData.shape)
        Test_Re = testData.drop(allList,axis = 1)
        if len(self.CatList) >= 1:
            enctransData = pd.DataFrame(self.enc.transform(testData[self.CatList]).toarray())
            Test_Re[self.nameList] = enctransData
        if len(self.ScaList) >= 1:
            scatransData = pd.DataFrame(self.scaler.transform(testData[self.ScaList]))
            Test_Re[self.ScaList] = scatransData
        print('Test_re shape: ', Test_Re.shape)
        return Test_Re
