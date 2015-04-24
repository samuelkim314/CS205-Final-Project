from mpi4py import MPI
import numpy as np
from sklearn.ensemble import RandomForestClassifier as Forest
import sklearn

class ForestParallel:
  rank = comm.Get_rank()
  size = comm.Get_size()
  
  def __init__(self, n_cores=1, n_estimators=10, criterion='gini'):
    self.n_cores = n_cores
    self.n_estimators = n_estimators
    self.criterion = criterior
    self.forest = Forest(n_estimators=n_estimators, criterion=criterion)
  
  def fit(X, y):
    self.forest.fit(X, y)
    if rank==0:
      self.estimators = comm.reduce(self.forest.estimators_)
    return self
