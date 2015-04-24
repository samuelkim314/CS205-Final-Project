from mpi4py import MPI
import numpy as np
from sklearn.ensemble import RandomForestClassifier as Forest
import sklearn

class ForestParallel:
  comm = MPI.COMM_WORLD  
  rank = comm.Get_rank()
  size = comm.Get_size()
  
  def __init__(self, n_cores=1, n_estimators=10, criterion='gini'):
    self.n_cores = n_cores
    self.n_estimators = n_estimators
    self.criterion = criterion
    self.forest = Forest(n_estimators=n_estimators, criterion=criterion)
  
  def fit(self, X, y):
    self.forest.fit(X, y)
    if self.rank==0:
      self.estimators = self.comm.reduce(self.forest.estimators_)
    return self
  
  def predict(self, X):
    #TODO: this isn't necessary if all cores get X
    if self.rank==0:
      X = self.comm.bcast(X)
    #TODO: need to broadcast the estimators
    predictions = self.forest.predict(X)
    predictions = self.comm.reduce(predictions)
    return predictions
  
  def predict_proba(self, X):
    X = comm.bcast(X)
    predictions = self.forest.predict_proba(X)
    predictions = comm.reduce(predictions)
    return predictions

if __name__ == '__main__':
  import pandas as pd
  
  #import data and labels
  train_data = pd.read_csv('train.csv')
  train_labels = pd.read_csv('train_label.csv')
  
  #separating dataset into training and testing for cross-validation
  test_idx = np.random.uniform(0, 1, len(train_data)) <= 0.1
  train = train_data[test_idx==True]
  trainLabels = train_labels[test_idx==True]
  test = train_data[test_idx==False]
  testLabels = train_labels[test_idx==False]
  
  features=['latitude','longitude']
  forest = ForestParallel(n_cores=2, n_estimators=10, criterion='gini')
  forest.fit(train[features],trainLabels['status_group'])
  predictions = forest.predict(test[features])
  print predictions[:10]

