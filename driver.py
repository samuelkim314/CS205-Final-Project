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

# Preprocessing: MRMR Parameters
p_rare = 0.01  # Discard factors rarer than this
p_mi = 0.95    # Accumulate factors up to this level of MI
middle_mrmr = False # Use MRMR to eliminate features early for speed
                    # at the cost of ~1% OOB accuracy   
   
if __name__ == '__main__':
    #Data import 
    train_data = pd.read_csv('train.csv')
    train_labels = pd.read_csv('train_label.csv')
    compete_data = pd.read_csv('test.csv')
    
    #Make a pd.Series with training labels
    label = train_labels.status_group
    #List of numerical features
    numer_feat = ['id', 'amount_tsh', 'date_recorded', 'gps_height', 'longitude', 'latitude', 'construction_year', 'population']
	
	#Work with copies of the data
    td = train_data.copy()
    cd = compete_data.copy()
    #Make a pd.Series with training labels
    label = train_labels.status_group
    
	#Preprocess the data and add rotated GPS coords to it
    for frame in [td, cd]:
        pp.initial_preprocess(frame)
        added_num_feat = pp.process_GPS(frame)
   
    numer_feat += added_num_feat
    
    #Vectorize the target variables    
    target = np.zeros(td.shape[0],dtype=np.int)
    target[np.array(label=='non functional')] = 0
    target[np.array(label=='functional needs repair')] = 1
    target[np.array(label=='functional')] = 2

    added_features = gGPS.generate_GPS(td, target, cd)
    numer_feat += added_features

	#Preprocessing begins here
# RAH: Added to parameters above so TFs and HT can see them more easily
#    p_rare = 0.01
#    p_mi = 0.95
    #Determines if you want to do MRMR for non-vectorized categorical features, or only after vectorization

# RAH: Added to top
#    #Switch this to False if you want more categorical features to boost oob_score by ~1%
#    middle_mrmr = False
    
    #Remove the values of categorical data which have occurrence less that p_rare
    pp.filter_rare_values(td, cd, numer_feat, p_rare)
    
    #Go through the categorical features and remove the features and values which deliver low information gain
	#(i.e., lower than (1-p_mi)*max_MI)
    irr_out = pp.filter_irrelevant_values(td, cd, numer_feat, label, p_mi)
      
    if middle_mrmr:
        #Now go through the remaining categorical features and retain only those which deliver 
        #large information gains and low redundance within the set
        mrmr_feat, mrmr_scores = pp.get_MRMR_features(td, numer_feat, label)
        pp.filter_MRMR_features(td, numer_feat, mrmr_feat)
        pp.filter_MRMR_features(cd, numer_feat, mrmr_feat)
    else:
        #The MRMR for entire features tends to eliminate a lot of features
        mrmr_feat = list( set(td.columns) - set(numer_feat))
	
    #Vectorize remaining features
    new_cfl = pp.one_hot(td, mrmr_feat)
    pp.one_hot(cd, mrmr_feat)
    
    #Now go through the vectorized variables one more time
    hot_mrmr_feat, hot_mrmr_scores = pp.get_MRMR_features(td, numer_feat, label)
    pp.filter_MRMR_features(td, numer_feat, hot_mrmr_feat)
    pp.filter_MRMR_features(cd, numer_feat, hot_mrmr_feat)

    #list the remaining categorical features 
    cfl_rem = list(set(td.columns) - set(numer_feat))
	
    #Transform data to matrix
    data_matrix = td.as_matrix().astype(np.float);
    data_matrix_compete = cd.as_matrix().astype(np.float);
    
    data_matrix.dump('processed_train_data')	
    data_matrix_compete.dump('processed_compete_data')
    target.dump('processed_targets')
	

    ##End of data preprocessing - next is forest creation and fitting
     
     
    #Here you can use numpy.load on the filenames specified before
    #Set up a forest predictor
    forest = Forest(n_estimators=200, criterion='gini', n_jobs=4, \
                    verbose=True, max_features='auto', bootstrap=True, \
                    min_samples_split=7, min_samples_leaf=1, oob_score=True)

    forest.fit(data_matrix,target)
	
    print forest.oob_score_
    
    predictions = forest.predict(data_matrix_compete)
    
	#Format the predictions to the submission requirements
    predictions_for_export = np.zeros(predictions.shape,dtype=np.object)
    predictions_for_export[predictions==0] = 'non functional'
    predictions_for_export[predictions==1] = 'functional needs repair'
    predictions_for_export[predictions==2] = 'functional'
    predictions_for_export = np.array([compete_data.id.values, predictions_for_export]).T
    
    np.savetxt("submit.csv",predictions_for_export,fmt='%s',delimiter=',',
               comments='',header='id,status_group')
    # RAH FIXME: comments='' should remove leading '# ' according to stackexchange
    #            but this is untested
    ##### REMEMBER: EDIT CSV FILE TO REMOVE HEADER'S LEADING # AND SPACE
