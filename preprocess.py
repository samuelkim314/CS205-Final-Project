import numpy as np
import pandas as pd
import sys
import re

# Definitions for manual data processing
def initial_preprocess(input_frame, irrelevant_features=None):
    # Convert date recorded to days since the first recording
    s = input_frame['date_recorded']
    s = s.apply(lambda date_string: np.datetime64(date_string))
    s = (s-s.min())/np.timedelta64(1,'D')
    input_frame.loc[:,'date_recorded']=s
    #Deal with region code
    input_frame['region_code'] = 'r'+input_frame['region_code'].astype(np.str) # Regions are categorical
    #Fill the missing values
    input_frame.fillna("was_nan",inplace=True)
    if irrelevant_features:
        for irfeat in irrelevant_features:
            input_frame.drop(irfeat, axis=1, inplace=True)

def rotate(coords,theta):
    out = np.zeros(coords.shape)
    out[:,0] = coords[:,0]*np.cos(theta) + coords[:,1]*np.sin(theta)
    out[:,1] = -coords[:,0]*np.sin(theta) + coords[:,1]*np.cos(theta)
    return out
    
def process_GPS(input_frame, rotation_number=5):
    added_feats = []
    XY = np.array([input_frame.longitude,input_frame.latitude]).T
    for i, theta in enumerate(np.linspace(0, np.pi/4, rotation_number)):
        if i>0:
            coords = rotate(XY, theta)
            lonr = 'longitude_r%d'%(i)
            latr = 'latitude_r%d'%(i)
            input_frame[lonr] = coords[:,0]
            input_frame[latr] = coords[:,1]
            added_feats.append(lonr)
            added_feats.append(latr)
            
    #Returnt the added numerical features
    return added_feats
            
def filter_rare_values(train_data_frame, compete_data_frame, numer_feat_list, freq_min=0.01):
    """The function analyzes the categorical features in train data to identify the rare elements and 
    replace corresponding entries by 'other'"""

    categ_feat_list = list(set(train_data_frame.columns) - set(numer_feat_list))
    irrelevant_features =[]
    for category in categ_feat_list:
        #print "Computing rare values for feature {}".format(category)
        #For each category, create a list of values to replace by "other" based on train data values
        train_series = train_data_frame.loc[:,category]
        compete_series = compete_data_frame.loc[:,category]
        rep_set = set(return_rare_values(train_series, freq_min))
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
            #print "Dropping feature {}\n".format(category)
        else:
            #Assign the modified series back to the data frame
            train_data_frame.loc[:,category] = train_series
            compete_data_frame.loc[:,category] = compete_series
            
    #print "\nAnalysis of feature rarity finished\n"

def return_rare_values(train_data_series, freq_min=0.01):
    
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
    
def compute_information_gain(feature_series, classification_series):
    """Function for figuring out the optimal number of distinct values to keep for each feature

    Input - Pandas Series with feature values, Pandas Series with classes

    Output - a Series indexed by distinct feature values, from the most abundant to the least abundant, and corresponding 
    mutual information with the classification data for the case when a given value and all above it 
    are considered as distinct, and the rest is lumped in "other" """

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
        #print f
        #Compute the mutual information
        output[f] = mutual_information(fl, clas)
        #Now replace in local copy of feature column all entries with this value to 'other'
        fl = fl.apply(lambda elem: 'other' if (elem == f) else elem)
    
    return output

def mutual_information(f1, f2):
    """The function accepts 2 CATEGORICAL feature columns from the same data frame and computes the mutual information for them
    
    When computing for class-feature mutual information, provide class as second argument
    In crosstab, i is the index of f1, j - of f2 (columns)
    """
    
    #First, make a crosstab with margins
    ct = pd.crosstab(f1, f2, margins=True)
    
    #Normalize it to get the frequencies
    N = ct['All']['All']
    ct /= N
    
    #Start computing the mutual information according to the literature prescription: double sum of terms like
    #rho_ij * log(rho_ij/(rho_i*rho_j))
    mi = 0.
    for col in ct.columns:
        if col is not 'All':            
            cc = ct[col] #Shorthand for entries corresponding to the current f2 category
            rho_j = cc['All'] #Probability of having a given f2 category entry
            for row in ct.index:
                #Loop over f1 categories
                if row is not 'All':
                    rho_ij = cc[row]
                    if rho_ij < 1E-13:
                        pass
                    else:
                        rho_i = ct.All[row]
                        mi += rho_ij * (np.log2(rho_ij / (rho_i * rho_j)))               
    return mi
            
def filter_irrelevant_values(train_data_frame, compete_data_frame, numer_feat_list, \
                                   classification_series, thr=0.9, mode='abs'):
    """Function for filtering out irrelevant values from information viewpoint based on correlation btw train data and class labels
    
    In rel mode, for each feature use the same percentage cutoff; in abs mode use 
    (1-thr) of the most informative feature as threshold
    
    Return - for each feature, a DataFrame having the values as index and columns MI and 'keep'
    having the mutual information for the case when a given value and those more frequent are kept 
    as separate, and the bool series explaining if the value is kept under current criteria"""
    
    #Sort the features according to mutual information
    categ_feat_list = list(set(train_data_frame.columns) - set(numer_feat_list))
    MI = pd.Series(np.zeros(len(categ_feat_list)), index=categ_feat_list)
    for feat in categ_feat_list:
        MI[feat] = mutual_information(train_data_frame[feat], classification_series)
        
    #Sort and take the largest mutual information
    MI.sort(ascending=False)
    #print "Ranking of features according to mutual information with labels in train_data"
    #print MI
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
    
    #In case the cutoff is expressed as a fixed percentage of the most informative feature
    #set the threshold accordingly
    if mode=='abs':
        #print "Absolute mode used, {}% of max MI is {}\n".format((1-thr)*100, delta)
        thr = delta
        
    #Iterate over remaining to remove irrelevant values there
    #Set up output dictionary indexed by the names of relevant features, and having a 
    #Frame with values, their information gains and if they are kept or not
    out_dict = {}
    for name in categ_feat_list:
        #print "Dealing with relevant feature {}".format(name)
        #Use the infer_irr_values to get the list of values to discard
        train_series = train_data_frame.loc[:,name]
        compete_series = compete_data_frame.loc[:,name]
        info_frame = analyze_value_information(train_series, classification_series, thr, mode)
        out_dict[name] = info_frame
        #Get the list of irrelevant values
        irrel_values = set(info_frame.index[info_frame.discard])
        #print "Value counts before plowing:\n"
        #print train_series.value_counts()
        #print "\nOut of them irrelevant are:\n"
        #print irrel_values
        
        #Change the irrelevant values to 'other'
        train_series = train_series.apply(lambda elem: 'other' if elem in irrel_values else elem)
        compete_series = compete_series.apply(lambda elem: 'other' if elem in irrel_values else elem)
        
        #Now check if some features can be discarded altogether after plowing
        rv = train_series.value_counts()
        #print "For feature {} informative values are:".format(name)
        #print rv
        if ( (rv.shape[0] == 1) or ((rv.shape[0] == 2) and ('other' in rv.index) and ('was_nan' in rv.index)) ):
            #Then the feature does not have any information at all
            train_data_frame.drop(name, axis=1, inplace=True)
            compete_data_frame.drop(name, axis=1, inplace=True)
            categ_feat_list.remove(name)
            #print "Dropping feature {}\n".format(name) 
        else:
            #Assign the modified series back to the data frames
            train_data_frame.loc[:,name] = train_series
            compete_data_frame.loc[:,name] = compete_series
            
    #Finally, return the dictionary storing the values discarded for each feature
    return out_dict

def analyze_value_information(feature_series, classification_series, threshold = 0.9, mode = 'abs'):
    """The mode can be either relative, discarding all features except first which push the information gain beyond thr*max,
    or absolute, when the values pushing the inf gain beyond max(inf_gain) - threshold"""
    
    fs = feature_series
    #Series with information gains, output is a Series (ins=INformation Series)
    ins = compute_information_gain(fs, classification_series)
    
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
    #Get it's index and swap the True to False there in above_thr
    swap_el = above_thr[above_thr == True].index[0]
    above_thr.loc[swap_el] = False
    #print above_thr
    #Create the output - Data Frame with information gains and bool labels for discarding the value or not
    output = pd.DataFrame({'MI':ins, 'discard':above_thr})
    #Now return the non-informative values
    return output
    
def filter_MRMR_features(input_frame, numer_feat_list, mrmr_feat_list):
    """Remove the categorical features irrelevant from MRMR perspective from a data frame"""
    
    
    categ_feat_set = set(input_frame.columns) - set(numer_feat_list)
    removal_list = list( categ_feat_set - set(mrmr_feat_list))
    #print "MRMR: removing {}\n\n".format(catfeat)
    input_frame.drop(removal_list, axis=1, inplace=True)


def get_MRMR_features(train_data_frame, numer_feat_list, classification_labels, mode='MIQ', cutoff=0.7):
    """Function returning the list of relevant and non-redundant features according to selected criterion
	
	The possible choices for mode include 
	mutual information quotient - MIQ: MI(candidate_feature, labels) / mean(MI(candidate_feature, selected_feat)), mutual information difference - MID: MI(candidate_feature, labels) - mean(MI(candidate_feature, selected_feat))
	hybrid - MIH: MI(candidate_feature, labels) * (MIQ - 1)
	The corresponding cutoffs need to be determined"""

    #Make a Series for mutual information with classification labels
    categ_feat_list = list(set(train_data_frame.columns) - set(numer_feat_list))
    MI = pd.Series(np.zeros(len(categ_feat_list)), index=categ_feat_list)
    for feat in categ_feat_list:
        #print feat
        MI[feat] = mutual_information(train_data_frame.loc[:,feat], classification_labels)
    #Sort and take the largest mutual information
    MI.sort(ascending=False)
    #print "MI with labels\n"
    #print MI

    #Set up a Series to hold the quotients for candidate variables at every step of the algorithm
    MQ = pd.Series(np.zeros(len(categ_feat_list)), index=categ_feat_list)

    #Put first the feature with highest mutual information to classification data
    #ac = accepted candidate
    ac = MI.index[0]
    MQ.drop(ac, inplace=True)
    #make output lists for added features, their scores and remaining features
    added = [ac]
    scores = [MI.iloc[0]]
    remain = list(set(categ_feat_list) - set(added))

    #Set up a table for mutual information between features; update as needed
    edata = np.empty((len(categ_feat_list), len(categ_feat_list))).astype(float)
    edata[:] = np.nan
    crossinf = pd.DataFrame(data=edata, columns=categ_feat_list, index=categ_feat_list)
    #Fill the table column-by-column for the already added features
    #Invariant: for all features in added, for all features in remain, crossinf[added][remain] != nan

    #print crossinf
    #Now keep adding the most relevant features to the output set
    while remain:
        #Compute the correlation between the last accepted feature and the remaining 
        for rem in remain:
            #print rem
            crossinf.loc[rem, ac] = mutual_information(train_data_frame.loc[:,ac], train_data_frame.loc[:,rem])
            #Now, for all remaining features, 
			#compute the relevant criterion based on mutual information with the classification
            #and the average mutual information with the accepted features
            #The former is a corresponding entry in MI series, and the latter is average of corresponding 
            #row of crossinf matrix
            if mode == 'MIQ':
				MQ[rem] = MI[rem] / crossinf.loc[rem, :].mean()
            elif mode == 'MID':
				MQ[rem] = MI[rem] - crossinf.loc[rem, :].mean()
            elif mode == 'MIH':
				MQ[rem] = MI[rem] * (MI[rem] / crossinf.loc[rem, :].mean() - 1)

        #New accepted candidate is the one with maximal entry in MQ    
        ac = MQ.argmax()
        score = MQ.max()
        #print "MQ\n", MQ
        #print "New candidate is {} with score {}\n\n".format(ac, score)
        if score < cutoff:
            #print "Omitting the last!\nStopping!\n"
            break

        #Update MQ and the sets
        MQ.drop(ac, inplace=True)
        added.append(ac)
        scores.append(score)
        remain.remove(ac)
        
    return added, scores
    
def one_hot(input_frame, relevant_features):
    
    new_feat_list = []
    for catfeat in relevant_features:
        for value in input_frame[catfeat].value_counts().index:
            #print value, catfeat
            new_name = str(catfeat) + '__' + str(value)
            new_feat_list.append(new_name)
            input_frame[new_name] = (input_frame[catfeat]==value)
        input_frame.drop(catfeat, axis=1, inplace=True)     
    return new_feat_list
    
#Definitions for wrapper approach

def permutation_importance(tree,test_data,test_target): # estimate variable importance using test data
    is_verbose = tree.get_params()['verbose']
    tree.set_params(verbose=False)
    importances = np.zeros(test_data.shape[1])
    original_score = tree.score(test_data,test_target)
    for i in xrange(test_data.shape[1]): # scramble each column and get % increase in error rate (Breinman importance)
        local = test_data.copy()
        np.random.shuffle(local[:,i])
        importances[i] = (original_score - tree.score(local,test_target))/(1-original_score)
        if is_verbose:
            sys.stdout.write('.')
            
    tree.set_params(verbose=is_verbose)
    return importances
