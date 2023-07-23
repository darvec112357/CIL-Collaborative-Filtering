import numpy as np
from surprise import AlgoBase

class UserAverage(AlgoBase):
    """
    Takes the average of all ratings given by a user
    """
    def __init__(self):
        AlgoBase.__init__(self)
    
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        self.global_mean = trainset.global_mean
        self.user_mean = dict() 
        for u, ratings in trainset.ur.items():
            self.user_mean[u] = np.mean([r for (_, r) in ratings])
        return self
    
    def estimate(self, u, i):
        if not (self.trainset.knows_user(u)):
            return self.trainset.global_mean
        return self.user_mean[u]
    
class ItemAverage(AlgoBase):
    """
    Takes the average of all ratings given to an item
    """
    def __init__(self):
        AlgoBase.__init__(self)
    
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        self.global_mean = trainset.global_mean
        self.item_mean = dict() 
        for i, ratings in trainset.ir.items():
            self.item_mean[i] = np.mean([r for (_, r) in ratings])
        return self
    
    def estimate(self, u, i):
        if not (self.trainset.knows_item(i)):
            return self.trainset.global_mean
        return self.item_mean[i]
    
class UserItemAverage(AlgoBase):
    """
    Takes the average of the average ratings given by a user and the average ratings given to an item
    """
    def __init__(self):
        AlgoBase.__init__(self)
    
    def fit(self, trainset):
        AlgoBase.fit(self, trainset)

        self.global_mean = trainset.global_mean
        self.user_mean = dict() 
        for u, ratings in trainset.ur.items():
            self.user_mean[u] = np.mean([r for (_, r) in ratings])
        self.item_mean = dict() 
        for i, ratings in trainset.ir.items():
            self.item_mean[i] = np.mean([r for (_, r) in ratings])
        return self
    
    def estimate(self, u, i):
        if not (self.trainset.knows_user(u) and self.trainset.knows_item(i)):
            return self.trainset.global_mean
        if not (self.trainset.knows_user(u)):
            return self.item_mean[i]
        if not (self.trainset.knows_item(i)):
            return self.user_mean[u]
        return (self.user_mean[u] + self.item_mean[i]) / 2