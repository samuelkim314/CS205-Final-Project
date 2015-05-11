"""Module with functions necessary for preprocessing categorical features

All public functions take one or 2 input frames and change it in-place.
Functions having 2 data frames in input arguments assume that feature names 
are the same.
Usage requires knowledge of names of numerical features in the 
dataset; this list is passed as parameter to most publci functions.
Processing has to be done in several steps:
1. initial_preprocess - fill in the missing values, convert 
    'date_recorded' into number of days since earliest recording,
    convert region code to categorical features.

2. filter_rare_values - infers which values have frequency < freq_min
    in training data, and replaces them with 'other' in both training 
    and test data

3. filter_irrelevant_values - infer which features in the data set
    contribute less than (1-thr) of maximal MI with classification labels
    among the features present in the dataset, and eliminate them.
    Do the same for values in remaining features.

4. get_MRMR_features - get the list of maximally relevant and minimally 
    redundant features among existing ones, based on training data.
    Can be applied before or after vectorization of categorical features.
   filter_MRMR_features - drop all categorical features except those
    deemed relevant by MRMR procedure in-place
"""

import numpy as np
import pandas as pd
import sys
import re

#USER INTERFACE FUNCTIONS

def initial_preprocess(input_frame, irrelevant_features=None):
    """Carry out dataset-specific feature handling
    
    In our case, fill in the missing values, convert 'date_recorded' 
    into number of days since earliest recording, optionally
    discard irrelevant features provided as input list.

    Args:
        input_frame     Pandas dataframe to modify in-place
        irrelevant_features     List of features to discard
    """

    #Convert date recorded to days since the first recording
    s = input_frame['date_recorded']
    s = s.apply(lambda date_string: np.datetime64(date_string))
    s = (s-s.min())/np.timedelta64(1,'D')
    input_frame.loc[:,'date_recorded']=s
    #Make region code categorical
    input_frame['region_code'] = 'r'+input_frame['region_code'].astype(np.str) # Regions are categorical
    #Fill the missing values
    input_frame.fillna("was_nan",inplace=True)
    if irrelevant_features:
        for irfeat in irrelevant_features:
            input_frame.drop(irfeat, axis=1, inplace=True)

def filter_rare_values(train_data_frame, compete_data_frame, numer_feat_list, freq_min=0.01):
    """Replace rare values in input frames with 'other'
    
    Computes the frequencies of distinct values of categorical features in the
    train_data_frame, and replace those below freq_min with 'other' in both
    train_data_frame and compete_data_frame.

    Args:
        train_data_frame    Pandas data frame for which the classification
                            labels exist; serves as base for freq counts
        compete_data_fame   Pandas data frame with competition  test data

        numer_feat_list     List of non-categorical features in above data
                            frames

        freq_min            Threshold for discarding values with freq<freq_min
                            in train_data_frame; both train and compete frames
                            are modified in-place
    """
    #Infer the categorical features in data frames
    categ_feat_list = list(set(train_data_frame.columns) - set(numer_feat_list))
    irrelevant_features =[]
    for category in categ_feat_list:
        #For each category, create a list of values to replace by "other" based on train data values
        train_series = train_data_frame.loc[:,category]
        compete_series = compete_data_frame.loc[:,category]
        #Use helper function to obtain list of rare values for this features
        rep_set = set(_return_rare_values(train_series, freq_min))
        #Now go through the data series to discard the rare values
        train_series = train_series.apply(lambda val: 'other' if val in rep_set else val)
        compete_series = compete_series.apply(lambda val: 'other' if val in rep_set else val)
        #Now check if one can discard the feature due to few unique values
        vc = train_series.value_counts()
        if ( (vc.shape[0] == 1) or ((vc.shape[0] == 2) and ('other' in vc.index) and ('was_nan' in vc.index)) ):
            #Then the feature does not have any information at all
            train_data_frame.drop(category, axis=1, inplace=True)
            compete_data_frame.drop(category, axis=1, inplace=True)
            irrelevant_features.append(category)
        else:
            #Assign the modified series back to the data frame
            train_data_frame.loc[:,category] = train_series
            compete_data_frame.loc[:,category] = compete_series

def filter_irrelevant_values(train_data_frame, compete_data_frame, numer_feat_list, \
                                   classification_series, thr=0.9, mode='abs'):
    """Filter out values and features based on MI with classification labels

    Filtering proceeds in several steps:
    1. MI with labels is computed for all features in train_data_frame,
    and features having MI < (1-thr) * MI_max are discarded
    2. For each feature, contributions to MI of that feature for its values are
    computed, and values below a certain threshold are replaced with 'other'. 
    Threshold is either (1-thr)*MI_max for 'abs' mode or (1-thr)*MI_curr for 'rel' mode

    Args:
        train_data_frame    Pandas data frame for which the classification
                            labels exist; serves as base for freq counts
        compete_data_fame   Pandas data frame with competition  test data

        numer_feat_list     List of non-categorical features in above data
                            frames

        classification_series   Pandas series with class labels for
                                train_data_frame entries

        thr                 Threshold for discarding features with MI <
                            (1-thr)*MI_max in train_data_frame; both train 
                            and compete frames are modified in-place

        mode = 'abs'/'rel'  Flag for setting the mode of filtering out values;
                            in 'abs' mode values are filtered out if they
                            contribute less than (1-thr) of MI for the most
                            informative feature in the train_data_frame; in
                            'rel' mode it is (1-thr) of MI for the feature under
                            consideration. In order to treat features and values
                            on equal footing (since features are vectorized),
                            usage of 'abs' mode is highly encouraged
    
    Output:
        dictionary with feature names as keys and DataFrames having the values
        of that feature as index, columns 'MI' with cumulative mutual information
        and 'discard' with bools indicating if the value is irrelevant
    """
    
    #Sort the features according to mutual information
    categ_feat_list = list(set(train_data_frame.columns) - set(numer_feat_list))
    MI = pd.Series(np.zeros(len(categ_feat_list)), index=categ_feat_list)
    for feat in categ_feat_list:
        MI[feat] = _mutual_information(train_data_frame[feat], classification_series)
        
    #Sort and take the largest mutual information
    MI.sort(ascending=False)
    largest_MI = MI[0]
    #Discard the features for which the mutual information is smaller than the loss in it 
    #for the feature with largest MI
    delta = (1 - thr) * largest_MI 
    #Go over the features and remove the irrelevant
    for feat, mi in MI.iteritems():
        #print MI, feat
        if mi < delta:
            categ_feat_list.remove(feat)
            train_data_frame.drop(feat, axis=1, inplace=True)
            compete_data_frame.drop(feat, axis=1, inplace=True)
            #print "Removing feature {} with MI {} smaller than {}".format(feat, mi, delta)
    
    #In case the cutoff is expressed as a fixed percentage of the most
    # informative feature, set the threshold for helper function accordingly
    if mode=='abs':
        thr = delta
        
    #Iterate over remaining features to remove irrelevant values there
    #Set up output dictionary indexed by the names of relevant features, and having a 
    #Frame with values, their information gains and if they are kept or not
    out_dict = {}
    for name in categ_feat_list:
        train_series = train_data_frame.loc[:,name]
        compete_series = compete_data_frame.loc[:,name]
        #Use the helper to get the list of values to discard
        info_frame = _analyze_value_information(train_series, classification_series, thr, mode)
        out_dict[name] = info_frame
        #Get the list of irrelevant values
        irrel_values = set(info_frame.index[info_frame.discard])
        #Change the irrelevant values to 'other'
        train_series = train_series.apply(lambda elem: 'other' if elem in irrel_values else elem)
        compete_series = compete_series.apply(lambda elem: 'other' if elem in irrel_values else elem)
        
        #Now check if some features can be discarded altogether after replacing
        #the values
        rv = train_series.value_counts()
        if ( (rv.shape[0] == 1) or ((rv.shape[0] == 2) and ('other' in rv.index) and ('was_nan' in rv.index)) ):
            #Then the feature does not have any information at all
            train_data_frame.drop(name, axis=1, inplace=True)
            compete_data_frame.drop(name, axis=1, inplace=True)
            categ_feat_list.remove(name)
        else:
            #Assign the modified series back to the data frames
            train_data_frame.loc[:,name] = train_series
            compete_data_frame.loc[:,name] = compete_series
            
    #Finally, return the dictionary storing the values discarded for each feature
    return out_dict


def get_MRMR_features(train_data_frame, numer_feat_list, classification_labels, mode='MIQ', cutoff=0.7):
    """Return the list of relevant and non-redundant features using MRMR filter
	
    Implementation is based on H. Peng's method. After selecting the feature
    with largest MI to labels, one keeps adding features based on maximizing MI
    to labels and minimizing MI with already selected features. More
    specifically, there are several ways to combine those quantities into one
    number:
	mutual information quotient - MIQ: 
        MI(candidate_feature, labels) / mean(MI(candidate_feature, selected_feat)) 
        
        mutual information difference - MID:
        MI(candidate_feature, labels) - mean(MI(candidate_feature, selected_feat))

    Preferred method is MIQ, based on literature results. Empirically, when MIQ
    for candidate features falls below .7, the obviously redundant features are
    selected in this particluar dataset. For MID mode the optimal cutoff was
    found to be 0.0.

    Args:
        train_data_frame    Pandas data frame for which the classification
                            labels exist

        numer_feat_list     List of non-categorical features in train data                        

        classification_labels   Pandas series with class labels for
                                train_data_frame entries

        mode = 'MIQ'/'MID'  Flag for setting the mode of choosing among
                            candidate features, see above

        cutoff              Stopping criterion for adding features; recommended
                            values are .7 for MIQ and 0.0 for MID
    
    Output:
        2-element tuple with list of feature names selected by the process and a
        list of MIQ/MIDs for them
    """

    #Make a Series for mutual information with classification labels
    categ_feat_list = list(set(train_data_frame.columns) - set(numer_feat_list))
    MI = pd.Series(np.zeros(len(categ_feat_list)), index=categ_feat_list)
    for feat in categ_feat_list:
        MI[feat] = _mutual_information(train_data_frame.loc[:,feat], classification_labels)
    #Sort and take the largest mutual information
    MI.sort(ascending=False)

    #Set up a Series to hold the quotients for candidate variables at every step of the algorithm
    MQ = pd.Series(np.zeros(len(categ_feat_list)), index=categ_feat_list)

    #Put first the feature with highest mutual information to classification data
    #ac = accepted candidate
    ac = MI.index[0]
    MQ.drop(ac, inplace=True)
    #Make output lists for added features, their scores and remaining features
    added = [ac]
    scores = [MI.iloc[0]]
    remain = list(set(categ_feat_list) - set(added))

    #Set up a table for mutual information between features; update as needed
    edata = np.empty((len(categ_feat_list), len(categ_feat_list))).astype(float)
    edata[:] = np.nan
    crossinf = pd.DataFrame(data=edata, columns=categ_feat_list, index=categ_feat_list)
    #Fill the table column-by-column for the already added features
    #Invariant: for all features in added, for all features in remain, crossinf[added][remain] != nan

    #Now keep adding the most relevant features to the output set
    while remain:
        #Compute the correlation between the last accepted feature and the remaining 
        for rem in remain:
            #print rem
            crossinf.loc[rem, ac] = _mutual_information(train_data_frame.loc[:,ac], train_data_frame.loc[:,rem])
            #Now, for all remaining features, 
            #compute the relevant criterion based on mutual information with the classification
            #and the average mutual information with the accepted features
            #The former is a corresponding entry in MI series, and the latter is average of corresponding 
            #row of crossinf matrix
            if mode == 'MIQ':
				MQ[rem] = MI[rem] / crossinf.loc[rem, :].mean()
            elif mode == 'MID':
				MQ[rem] = MI[rem] - crossinf.loc[rem, :].mean()
        #New accepted candidate is the one with maximal entry in MQ    
        ac = MQ.argmax()
        score = MQ.max()
        #print "New candidate is {} with score {}\n\n".format(ac, score)
        if score < cutoff:
            break

        #Update MQ and the sets
        MQ.drop(ac, inplace=True)
        added.append(ac)
        scores.append(score)
        remain.remove(ac)
        
    return added, scores
    
def filter_MRMR_features(input_frame, numer_feat_list, mrmr_feat_list):
    """Remove the categorical features irrelevant from MRMR perspective from a data frame
    
    Forms a list of categorical features in the input_frame, and then drops
    inplace all those not in to mrmr_feat_list

    Args:
        input_frame         Pandas data frame to filter inplace

        numer_feat_list     List of non-categorical features in input frame

        mrmr_feat_list      List of features obtained by running mrmr filtration
    """
    categ_feat_set = set(input_frame.columns) - set(numer_feat_list)
    removal_list = list( categ_feat_set - set(mrmr_feat_list))
    input_frame.drop(removal_list, axis=1, inplace=True)

def one_hot(input_frame, relevant_features):
    """Vectorize the relevant_features in input_frame inplace
    
    For each value in relevant_features create a bool column, and then drop that
    feature column
    """
    new_feat_list = []
    for catfeat in relevant_features:
        for value in input_frame[catfeat].value_counts().index:
            new_name = str(catfeat) + '__' + str(value)
            new_feat_list.append(new_name)
            input_frame[new_name] = (input_frame[catfeat]==value)
        input_frame.drop(catfeat, axis=1, inplace=True)     
    return new_feat_list

#HELPER FUNCTIONS
def _return_rare_values(train_data_series, freq_min=0.01):
    """Helper; return list of rare values in input series
    
    Computes the frequencies of distinct values of train_data_series, 
    and return a list of names of values occuring rare than freq_min;
    also, consolidate all 'other' values into one.

    Args:
        train_data_series   Pandas data series for analysis

        freq_min            Threshold for returning names of values with freq<freq_min
                            in train_data_series
    """
    
    vc = train_data_series.value_counts()
    #Get the names of values which correspond to frequency smaller than freq_min 
    vcr = vc/vc.sum()
    rare_list = list(vcr[vcr<freq_min].index)
    
    #Figure out if "other" categories are present in the list
    other_list = []
    reother = re.compile('^other', re.IGNORECASE)
    for elem in vcr.index[:]:
        if len(str(elem)) >= 5:
            if reother.match(str(elem)):
                other_list.append(elem)
    #Overall list of values to replace
    replist = other_list + rare_list
    return replist
    
def _compute_information_gain(feature_series, classification_series):
    """Helper; infer the contributions to MI with labels for each value
    
    Replace rare values in input frames with 'other'
    
    Computes the frequencies of distinct values of categorical features in the
    train_data_frame, and replace those below freq_min with 'other' in both
    train_data_frame and compete_data_frame.

    Args:
        feature_series      Pandas Series to analyze for mutual information with
                            available labels
        
        classification_series   Pandas Series holding classification labels for
                                the first argument

    Output:
        Pandas Series indexed by distinct feature values, from the most to the least 
        frequent, and corresponding mutual information with the classification data 
        for the case when a given value and all more frequent are considered as distinct,
        and the rest is lumped in "other" 
    """

    #Make a local copy of the feature Series
    fl = feature_series.copy()
    clas = classification_series

    #First - get the unique values for the feature of interest, drop the "other" column from it
    dv = fl.value_counts()
    if 'other' in list(dv.index):
        dv.drop('other', inplace=True)
    #Make a stab for the output Series
    output = dv.copy().astype('float')
    output.name = fl.name

    #Sort to iterate over variables in ascending order or abundance
    dv.sort()
    #Now go over non-other features one-by-one, from the least significant, and compute the mutual information including it and more abundant features
    for f in dv.index:
        #Compute the mutual information
        output[f] = _mutual_information(fl, clas)
        #Now replace in local copy of feature column all entries with this value to 'other'
        fl = fl.apply(lambda elem: 'other' if (elem == f) else elem)
    return output

def _mutual_information(series_1, series_2):
    """Helper; compute MI for input data series

    Args:
        series_1, series_2      Pandas data series of the same length

    Output:
        float with value of mutual information (MI)
    """
    
    #First, make a crosstab with axis sums (margins)
    ct = pd.crosstab(series_1, series_2, margins=True)
    
    #Normalize it to get the frequencies
    N = ct['All']['All']
    ct /= N
    
    #Start computing the mutual information according to the literature prescription: double sum of terms like
    #rho_ij * log(rho_ij/(rho_i*rho_j))
    mi = 0.
    for col in ct.columns:
        if col is not 'All':            
            cc = ct[col] #Shorthand for entries corresponding to the current series_2 category
            rho_j = cc['All'] #Probability of having a given series_2 category entry
            for row in ct.index:
                #Loop over series_1 categories
                if row is not 'All':
                    rho_ij = cc[row]
                    if rho_ij < 1E-13:
                        pass
                    else:
                        rho_i = ct.All[row]
                        mi += rho_ij * (np.log2(rho_ij / (rho_i * rho_j)))               
    return mi

def _analyze_value_information(feature_series, classification_series, threshold = 0.9, mode = 'abs'):
    """Helper, compute cumulative MI for a feature, infer which values to keep
    
    After computing MI, decides on discarding the values in 2 modes: absolute
    ('abs')  discards all value except first with cumulative MI within threshold from
    total MI for this feature; relative ('rel') discards all values except first
    with cumulative MI higher then threshold * MI of current feature. The
    rationale for keeping the first value is that it can have substantial
    information gain of its own.

    Args:
        feature_series      Pandas series with feature under consideration

        classification_series   Pandas series with labels to compute MI
                                with the input feature

        mode='abs'/'rel'    Flag for switching mode of filtration; 'abs' to cut
                            off values based on input of certain MI values;
                            'rel' to cut off values based on maximal MI of a
                            given feature

        threshold           MI value in 'abs' mode, fraction of features's max
                            MI in 'rel' mode
    """
    fs = feature_series
    #Series with information gains, output is a Series (ins=INformation Series)
    ins = _compute_information_gain(fs, classification_series)
    #For each entry of the feature series, check if the relevance of the value is below the threshold;
    #if not, replace the entry by "other"
    #Get the absolute threshold
    if mode=='rel':
        thr = threshold * ins.max()
    elif mode=='abs':
        thr = ins.max() - threshold
    #Series with True if we want to discard the value
    above_thr = ins > thr    
    #Keep the first value crossing the threshold, too
    #Get its index and swap the True to False there in above_thr
    swap_el = above_thr[above_thr == True].index[0]
    above_thr.loc[swap_el] = False
    #Create the output - Data Frame with information gains and bool labels for discarding the value or not
    output = pd.DataFrame({'MI':ins, 'discard':above_thr})
    return output
