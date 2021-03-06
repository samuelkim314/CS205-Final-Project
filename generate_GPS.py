import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as knn_classifier

# Rotates the given coordinates in 2D by angle theta
#
# coords - Two-column numpy array of floating-point values.
#          Each row is considered a coordinate pair.
# theta - Floating point scalar
#
# Returns rotated coordinates in an array with the same shape as coords.
def rotate(coords,theta):
    out = np.zeros(coords.shape)
    out[:,0] = coords[:,0]*np.cos(theta) + coords[:,1]*np.sin(theta)
    out[:,1] = -coords[:,0]*np.sin(theta) + coords[:,1]*np.cos(theta)
    return out

# Sums up the inverse distance to neighbors, grouped by class.
#
# This function calculates the inverse distance to each neighbor,
# then sums the values corresponding to each class.
#
# distances - Floating-point numpy array of shape num_samples x num_neighbors
#             as returned by sklearn.neighbors.KNeighborsClassifier.kneighbors
#             with return_distance=True.
# neighbors - An integer numpy array of shape num_samples x num_neighbors
#             as returned by sklearn.neighbors.KNeighborsClassifier.kneighbors
# classes - An integer numpy array of shape num_samples with elements
#           equal to 0, 1, or 2.
#
# Returns a numpy array of shape num_samples x 3, where return_array[i,j]
# is the sum of inverse distances to all of sample i's neighbors that have
# class j.
def inverse_distance_class(distances,neighbors,classes):
    d = distances.copy()
    d[d==0] = np.inf
    d = 1.0/d
    class_sums = np.zeros((distances.shape[0],3))
    for i in range(distances.shape[0]):
        for c in range(3):
            class_sums[i,c] = np.sum(distances[i,classes[neighbors[i]]==c])
    return class_sums

# Sums up a sigmoid function of distance to neighbors, grouped by class.
#
# This function calculates a sigmoid function of the distance to each neighbor,
# then sums the values corresponding to each class.
#
# distances - Floating-point numpy array of shape num_samples x num_neighbors
#             as returned by sklearn.neighbors.KNeighborsClassifier.kneighbors
#             with return_distance=True.
# neighbors - An integer numpy array of shape num_samples x num_neighbors
#             as returned by sklearn.neighbors.KNeighborsClassifier.kneighbors
# classes - An integer numpy array of shape num_samples with elements
#           equal to 0, 1, or 2.
#
# Returns a numpy array of shape num_samples x 3, where return_array[i,j]
# is the sum of sigmoid functions of distance to all of sample i's neighbors 
# that have class j.
def sigmoid_distance_class(distances,neighbors,classes):
    class_sums = np.zeros((distances.shape[0],3))
    mu = 0.001
    kt = 0.001
    sig = distances/(1+np.exp((distances-mu)/kt))
    for i in range(neighbors.shape[0]):
        for c in range(3):
            class_sums[i,c] = np.sum(sig[i,classes[neighbors[i]]==c])
    return class_sums

# Sums up a sigmoid function of distance to neighbors, grouped by class.
#
# This function calculates a sigmoid function of the distance to each neighbor,
# then sums the values corresponding to each class.
# Very similar to sigmoid_distance_class, but the range of the sigmoid
# is longer.
#
# distances - Floating-point numpy array of shape num_samples x num_neighbors
#             as returned by sklearn.neighbors.KNeighborsClassifier.kneighbors
#             with return_distance=True.
# neighbors - An integer numpy array of shape num_samples x num_neighbors
#             as returned by sklearn.neighbors.KNeighborsClassifier.kneighbors
# classes - An integer numpy array of shape num_samples with elements
#           equal to 0, 1, or 2.
#
# Returns a numpy array of shape num_samples x 3, where return_array[i,j]
# is the sum of sigmoid functions of distance to all of sample i's neighbors 
# that have class j.
def long_sigmoid_distance_class(distances,neighbors,classes):
    class_sums = np.zeros((distances.shape[0],3))
    mu = 0.001
    kt = 0.007
    sig = distances/(1+np.exp((distances-mu)/kt))
    for i in range(neighbors.shape[0]):
        for c in range(3):
            class_sums[i,c] = np.sum(sig[i,classes[neighbors[i]]==c])
    return class_sums

# Determines the number of neighbors of each class for all given samples.
#
# This function groups neighbors to each input sample by class, and returns
# the corresponding number of neighbors of that class.
#
# distances - Floating-point numpy array of shape num_samples x num_neighbors
#             as returned by sklearn.neighbors.KNeighborsClassifier.kneighbors
#             with return_distance=True. Not used, but it's an argument
#             to keep the interface identical to the other *_class functions.
# neighbors - An integer numpy array of shape num_samples x num_neighbors
#             as returned by sklearn.neighbors.KNeighborsClassifier.kneighbors
# classes - An integer numpy array of shape num_samples with elements
#           equal to 0, 1, or 2.
#
# Returns a numpy array of shape num_samples x 3, where return_array[i,j]
# is the number of sample i's neighbors that have class j.
def num_neighbors_class(distances,neighbors,classes):
    class_sums = np.zeros((neighbors.shape[0],3))
    for i in range(neighbors.shape[0]):
        for c in range(3):
            class_sums[i,c] = np.sum(classes[neighbors[i]]==c)
    return class_sums

# List of class-based functions and their corresponding text names.
generators = [inverse_distance_class, sigmoid_distance_class,
              long_sigmoid_distance_class, num_neighbors_class]
generator_names = ['inv_dist','sigmoid','long_sigmoid',
                   'NN']

# Appends GPS data-based columns to the Pandas frames
# for training and testing data
#
# train_frame, test_frame: Pandas data frames
# train_classes: integer numpy vector of classes whose elements
#                are equal to 0, 1, or 2
# num_rotations: integer, number of rotations of GPS data to use
#                Includes 0 degrees (e.g. num_rotations==1
#                does nothing, 2 appends a single rotation)
# num_neighbors: integer, functions of nearest neighbors will use
#                this many total neighbors
#
# Returns appended_columns, a list of column names that
#                were appended
def generate_GPS(train_frame, train_classes, test_frame, 
                 num_rotations = 5, num_neighbors = 30):

    appended_columns = []

    # Simple rotations of GPS data
    for frame in [test_frame, train_frame]:
        XY = np.c_[frame.longitude, frame.latitude]
        for i,theta in enumerate(np.linspace(0,np.pi/4,num_rotations)):
            if i>0:
                rotated = rotate(XY,theta)
                frame['longitude_%d'%(i)] = rotated[:,0]
                frame['latitude_%d'%(i)] = rotated[:,1]
    
    for i in range(1, num_rotations):
        appended_columns += ['longitude_%d'%(i), 'latitude_%d'%(i)]

    # Sum of a function of distances over the class
    knn = knn_classifier(n_neighbors = 30)
    knn.fit(XY, train_classes)
    
    # Compute functions for training data
    dists,neighbors = knn.kneighbors(return_distance=True)
    for i, generator in enumerate(generators):
        location_data = generator(dists, neighbors, train_classes)
        for j in range(3):
            name = '%s_%d'%(generator_names[i], j)
            train_frame[name] = location_data[:,j]
            appended_columns += [name]

    # Compute functions for testing data, using the nearest
    # neighbors from the training data (XY is left over from above)
    XYt = np.c_[test_frame.longitude, test_frame.latitude]
    dists,neighbors = knn.kneighbors(XYt, return_distance=True)
    for i, generator in enumerate(generators):
        location_data = generator(dists, neighbors, train_classes)
        for j in range(3):
            name = '%s_%d'%(generator_names[i], j)
            test_frame[name] = location_data[:,j]

    return appended_columns

if __name__ == '__main__':
    # Load data
    train_data = pd.read_csv('train.csv')
    train_labels = pd.read_csv('train_label.csv')
    compete_data = pd.read_csv('test.csv')
    compete_id = compete_data.id
    N = train_data.shape[0]
    # generate class integers    
    target = np.zeros(N,dtype=np.int)
    target[np.array(train_labels['status_group']=='non functional')] = 0
    target[np.array(train_labels['status_group']=='functional needs repair')] = 1
    target[np.array(train_labels['status_group']=='functional')] = 2
    
    # Produce and append rotated GPS and kNN-derived data columns
    # to the training data and competition data frames.
    generate_GPS(train_data, target, compete_data)
