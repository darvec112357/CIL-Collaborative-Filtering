# -*- Encoding:UTF-8 -*-

import numpy as np
import sys
import pandas as pd

class DataSet(object):
    def __init__(self, fileName):
        self.data, self.shape = self.getData(fileName)
        self.train, self.test = self.getTrainTest()
        self.trainDict = self.getTrainDict()

    def reformat_id(self, id):
        # split and reformat the df
        row, col = id.split('_')
        return int(row[1:]), int(col[1:])

    def _read_df_in_format(self,root):
        df = pd.read_csv(root)
        df['row'], df['col'] = zip(*df['Id'].map(self.reformat_id))
        df.drop('Id', axis=1, inplace=True)
        return df

    def getData(self, fileName):
        print("Loading data_train...")
        data = []
        u = 0
        i = 0
        maxr = 0.0
        # df = self._read_df_in_format(fileName)
        df = pd.read_csv('data_train_small.csv')
        columns = ['row','col','Prediction']
        df = df[columns]
        for i in range(len(df)):
            if i%100000==0 :
                print(i)
            data.append(tuple(df.iloc[i]))
        u = df['row'].max()
        i = df['col'].max()
        maxr = df['Prediction'].max()
        self.maxRate = maxr
        print("Loading Success!\n"
                "Data Info:\n"
                "\tUser Num: {}\n"
                "\tItem Num: {}\n"
                "\tData Size: {}".format(u, i, len(data)))
        return data, [u, i]

    def getTrainTest(self):
        data = self.data
        train = []
        test = []
        for i in range(len(data)-1):
            user = data[i][0]-1
            item = data[i][1]-1
            rate = data[i][2]
            if data[i][0] != data[i+1][0]:
                test.append((user, item, rate))
            else:
                train.append((user, item, rate))

        test.append((data[-1][0]-1, data[-1][1]-1, data[-1][2]))
        return train, test

    def getTrainDict(self):
        dataDict = {}
        for i in self.train:
            dataDict[(i[0], i[1])] = i[2]
        return dataDict

    def getEmbedding(self):
        train_matrix = np.zeros([self.shape[0], self.shape[1]], dtype=np.float32)
        for i in self.train:
            user = i[0]
            movie = i[1]
            rating = i[2]
            train_matrix[user][movie] = rating
        return np.array(train_matrix)

    def getInstances(self, data, negNum):
        user = []
        item = []
        rate = []
        for i in data:
            user.append(i[0])
            item.append(i[1])
            rate.append(i[2])
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (i[0], j) in self.trainDict:
                    j = np.random.randint(self.shape[1])
                user.append(i[0])
                item.append(j)
                rate.append(0.0)
        return np.array(user), np.array(item), np.array(rate)

    def getTestNeg(self, testData, negNum):
        user = []
        item = []
        for s in testData:
            tmp_user = [s[0]]
            tmp_item = [s[1]]
            neglist = set([s[1]])
            for t in range(negNum):
                j = np.random.randint(self.shape[1])
                while (s[0], j) in self.trainDict or j in neglist:
                    j = np.random.randint(self.shape[1])
                neglist.add(j)
                tmp_user.append(s[0])
                tmp_item.append(j)
            user.append(tmp_user)
            item.append(tmp_item)
        return [np.array(user), np.array(item)]
