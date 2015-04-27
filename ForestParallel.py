from mpi4py import MPI
import numpy as np
from sklearn.ensemble import RandomForestClassifier as Forest
import sklearn

class ForestParallel:
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  estimators = []
  
  def __init__(self, n_cores=1, n_estimators=10, criterion='gini'):
    self.n_cores = n_cores
    self.n_estimators = n_estimators
    self.criterion = criterion
    self.forest = Forest(n_estimators=n_estimators, criterion=criterion)
  
  def fit(self, X, y):
    #distribute fitting task and gather all the estimators to all cores
    self.forest.fit(X, y)
    #TODO: decide between gather and allgather
    self.estimators = self.comm.allgather(self.forest.estimators_)
    #flatten list
    self.estimators = [tree for sublist in self.estimators for tree in sublist]
    self.forest.estimators_ = self.estimators
    return self
  
  def predict(self, X):
    #predictions on just one core
    if self.rank==0:
      predictions = self.forest.predict(X)
      return predictions
    return None
  
  def predictPar(self, X):
    #predictions using all the cores
    #TODO: Finish
    if self.rank==0:
      estimators = self.comm.scatter(self.forest.estimators_)
    self.forest.estimators_ = estimators
    predictions = self.forest.predict(X)
    if self.rank==0:
      predictions = self.comm.gather(predictions)
      print predictions.shape
  
  def predict_proba(self, X):
    #probability predictions on one core
    if self.rank==0:
      predictions = self.forest.predict_proba(X)
      return predictions
    return None

def getData():
  #imports data and labels to all cores
  import pandas as pd
  
  comm = MPI.COMM_WORLD
  rank = comm.Get_rank()
  size = comm.Get_size()
  
  #buffer for data to be broadcast
  train = None
  trainLabels = None
  test = None
  testLabels = None
  
  if rank==0:
    #import data and labels
    train_data = pd.read_csv('train.csv', index_col='id')
    train_labels = pd.read_csv('train_label.csv', index_col='id')
    
    #separating dataset into training and testing for cross-validation
    test_idx = np.random.uniform(0, 1, len(train_data)) <= 0.9
    features=['latitude','longitude']
    train = train_data[test_idx==True][features]
    trainLabels = train_labels[test_idx==True]
    test = train_data[test_idx==False][features]
    testLabels = train_labels[test_idx==False]

  train = comm.bcast(train, root=0)
  trainLabels = comm.bcast(trainLabels, root=0)
  test = comm.bcast(test, root=0)
  testLabels = comm.bcast(testLabels, root=0)
  
  return train, trainLabels, test, testLabels

if __name__ == '__main__':
  train, trainLabels, test, testLabels = getData()
  size = MPI.COMM_WORLD.Get_size()
  
  forest = ForestParallel(n_cores=size, n_estimators=10, criterion='gini')
  forest.fit(train,trainLabels['status_group'])
  
  predictions = forest.predict(test[features])
  print "done"
  #if MPI.COMM_WORLD.Get_rank()==0:
  #  print predictions.shape
  #print predictions[:10]

