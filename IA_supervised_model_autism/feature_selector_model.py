
# Load the function to performe feature selection
from sklearn.feature_selection import SelectKBest
#load the feature selection algorithms
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import mutual_info_regression
from sklearn.feature_selection import SelectPercentile
from sklearn.feature_selection import SelectFpr
from sklearn.feature_selection import SelectFdr
from sklearn.feature_selection import SelectFwe
from sklearn.feature_selection import GenericUnivariateSelect

class FeatureSelectorModel:
    
    def __init__(self) -> None:
        pass
    
    # function to do feature selection
    def feature_selector(self, X_train,y_train,X_test,type,i):
        if (type == 1):
    #ANOVA F-value between label/feature for classification tasks.
            bestfeatures = SelectKBest(score_func = f_classif, k=i)
        elif(type == 2):
    #Mutual information for a discrete target.
            bestfeatures = SelectKBest(score_func=mutual_info_classif, k=i)
        elif(type == 3):
        #Chi-squared stats of non-negative features for classification tasks.
            bestfeatures = SelectKBest(score_func=chi2, k=i)
        elif(type == 4):
    #Select features based on an estimated false discovery rate.
            bestfeatures = SelectKBest(score_func=SelectFdr, k=i)
        elif(type == 5):
    #Select features based on family-wise error rate.
            bestfeatures = SelectKBest(score_func=SelectFwe, k=i)
    #Perform the feature based on selected algorithm
        fit = bestfeatures.fit(X_train,y_train)
        cols_idxs = fit.get_support(indices=True)
        Xt=X_train.iloc[:,cols_idxs] # extract the best features for training
        Xteste=X_test.iloc[:,cols_idxs] # extract the best features for testing
        return Xt,Xteste