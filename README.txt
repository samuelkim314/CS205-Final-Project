-------------
Title
-------------
Predicting Water Well Statuses in Tanzania
CS 205 Final Project
Dmitry Vinichenko, Sam Kim, Robert Hoyt

-------------
Usage
-------------

driver.py
  Example: python driver.py
  
driver_parallel.py
  Format: mpirun -n n_cores python driver_parallel.py total
  Arguments:
    n_cores: number of cores to use
    total:  total number of trees in the random forest
  Example: mpirun -n 4 python driver_parallel.py 2048
  
driver_parallel_ms.py
  Format: mpirun -n n_cores python driver_parallel.py total each
  Arguments:
    n_cores: number of cores to use
    total:  total number of trees in the random forest
    each:   number of trees in a single work unit for load balancing
  Example: mpirun -n 4 python driver_parallel.py 2048 8

-------------
Dependencies
-------------
sklearn requires at least version 0.16.1
  sklearn 15 and below has a different interface for KNeighbors
  
-------------
Contents:
-------------

driver.py
  Imports the raw training data, training labels, and test data, processes the data and extracts features, trains random forests on the training data, and makes predictions on the test data using the random forests.

driver_parallel.py:
  Same as driver.py, but implemented in parallel.
  
driver_parallel_ms.py:
  Same as driver_parallel.py, but implements load-balancing using master-slave configuration.
  
ForestParallel.py:
  Class that implements training random forests in parallel.


