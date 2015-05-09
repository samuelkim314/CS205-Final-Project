from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.ensemble import RandomForestClassifier as Forest
from sklearn.ensemble import ExtraTreesClassifier as EForest
from sklearn.cross_validation import train_test_split as sk_split
from sklearn.neighbors import KNeighborsClassifier as KNN
import generate_GPS as gGPS
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import re
import preprocess as pp

from ForestParallel import ForestParallel as ForestPar
from mpi4py import MPI
    
if __name__ == '__main__':
    # Get num trees from command line
    try:
      total = int(sys.argv[1])
    except:
      print 'Usage: mpirun -n n_cores python driver_benchmark_even.py total'
      sys.exit()
    
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    data_matrix = None
    data_matrix_compete = None
    target = None
    
    if rank==0:
      data_matrix = np.load('processed_train_data')
      data_matrix_compete = np.load('processed_compete_data')
      target = np.load('processed_targets')
    
    #broadcasting data
    p_start = MPI.Wtime()
    data_matrix = comm.bcast(data_matrix, root=0)
    target = comm.bcast(target, root=0)
    comm.barrier()
    p_stop = MPI.Wtime()
    data_btime = p_stop - p_start  #broadcast time for training data
    if rank==0:
      print "Broadcast time: " + str(data_btime)
     
    #Here you can use numpy.load on the filenames specified before
    #Set up a forest predictor
    """forest = Forest(n_estimators=200, criterion='gini', n_jobs=4, \
                    verbose=True, max_features='auto', bootstrap=True, \
                    min_samples_split=7, min_samples_leaf=1, oob_score=True)"""
    if rank==0:
      print "Cores, total trees, trees per core, time"
    #for total in [512, 1024, 2048]:
    #  for each in [1, 2, 4, 8, 16, 32, 64, 128]:
    p_start = MPI.Wtime()
    forest = ForestPar(n_estimators=total, criterion='gini', \
                min_samples_split=7)

    forest.fit(data_matrix,target)
    p_stop = MPI.Wtime()
    runtime = p_stop - p_start  #broadcast time for training data
    if rank==0:
      print size, total, runtime

