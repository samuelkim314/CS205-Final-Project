from sklearn.tree import DecisionTreeClassifier as Tree
from sklearn.ensemble import RandomForestClassifier as Forest
from sklearn.ensemble import ExtraTreesClassifier as EForest
from sklearn.cross_validation import train_test_split as sk_split
from sklearn.neighbors import KNeighborsClassifier as KNN
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
import re
import preprocess as pp


    
if __name__ == '__main__':
    #Data import 
    train_data = pd.read_csv('train.csv')
    train_labels = pd.read_csv('train_label.csv')
    compete_data = pd.read_csv('test.csv')
    
    N = train_data.shape[0]
    
    train_data.fillna("was_nan",inplace=True)
    compete_data.fillna("was_nan",inplace=True)

    
    #Make a pd.Series with training labels
    label = train_labels.status_group
    #List of categ features
    all_feat = list(train_data.columns)
    #Irrelevant features
    irr_feat=[]
    #List of numerical features
    numer_feat = ['id', 'amount_tsh', 'date_recorded', 'gps_height', 'longitude', 'latitude', 'construction_year', 'population']
    #Get the list of categorical features 
    cfl = list(set(all_feat) - set(numer_feat) - set(irr_feat))
    
        
    #Preprocess the data and add rotated GPS coords to it
    ntest = 400#N
    
    td = train_data[:ntest].copy()
    cd = compete_data[:ntest].copy()
    label = train_labels.status_group[:ntest].copy()
    
    for frame in [td, cd]:
        pp.preprocess(frame, irr_feat)
        added_num_feat = pp.process_GPS(frame)
    
    numer_feat += added_num_feat
    cfl = list( set(cfl) - set(added_num_feat))
    
    #Vectorize the target variables
    
    target = np.zeros(ntest,dtype=np.int)
    target[np.array(label=='non functional')] = 0
    target[np.array(label=='functional needs repair')] = 1
    target[np.array(label=='functional')] = 2

    p_rare = 0.01
    p_mi = 0.98
    mode_mi = 'abs'
    p_mi_hot = 0.98
    mode_mi_hot = 'abs'
    
    #Weed out the values of categorical data which have occurence less that p_rare
    pp.filter_rare_values(td, cd, numer_feat, p_rare)
    
    #Go through the categ features and remove the values which deliver low information gain
    irr_out = pp.filter_irrelevant_values(td, cd, numer_feat, label, p_mi, mode_mi)
            
    #Now go through the remaining categorical features and retain only those which deliver 
    #large information gains and low redundance within the set
    mrmr_feat, mrmr_scores = pp.get_MRMR_features(td, numer_feat, label)
    pp.filter_MRMR_features(td, numer_feat, mrmr_feat)
    pp.filter_MRMR_features(cd, numer_feat, mrmr_feat)
    #Only informative features remain, vectorizes them as store names as the new categorical features
    new_cfl = pp.one_hot(td, mrmr_feat)
    pp.one_hot(cd, mrmr_feat)
    
    #Now weed through the vectorized variables one more time
    hot_mrmr_feat, hot_mrmr_scores = pp.get_MRMR_features(td, numer_feat, label)
    pp.filter_MRMR_features(td, numer_feat, hot_mrmr_feat)
    pp.filter_MRMR_features(cd, numer_feat, hot_mrmr_feat)
    pass

    #Transform data to matrix
    data_matrix = td.as_matrix().astype(np.float);
    data_matrix_compete = cd.as_matrix().astype(np.float);
    
    #Prepare for cross-validation
    data_train, data_test, target_train, target_test = sk_split(data_matrix, target, test_size=0.1)
     
    #Set up a forest predictor
    forest = Forest(n_estimators=20, criterion='gini', n_jobs=4, \
                    verbose=True, max_features=10, bootstrap=True, \
                    min_samples_split=7, min_samples_leaf=1, oob_score=False)

    forest.fit(data_matrix,target)
    
    predictions = forest.predict(data_matrix_compete)
    
    predictions_for_export = np.zeros(predictions.shape,dtype=np.object)
    predictions_for_export[predictions==0] = 'non functional'
    predictions_for_export[predictions==1] = 'functional needs repair'
    predictions_for_export[predictions==2] = 'functional'
    predictions_for_export = np.array([compete_data.id.values, predictions_for_export]).T
    
    np.savetxt("submit.csv",predictions_for_export,fmt='%s',delimiter=',',header='id,status_group')
    ##### REMEMBER: EDIT CSV FILE TO REMOVE HEADER'S LEADING # AND SPACE