#for numerical coputing
import numpy as np

#for dataforms
import pandas as pd

#import warnings
import warnings as wng
wng.filterwarnings("ignore")

import matplotlib.pyplot as plt #toolbox to do plots

# To split train and test sets
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC #load the support vector machine model functions
from sklearn.metrics import accuracy_score

# Machine Learn Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import classification_report # get the report
from sklearn import preprocessing # normalize the features
from sklearn.preprocessing import MinMaxScaler # normalize the features
from sklearn.feature_selection import SelectKBest #load the feature selector model  
from sklearn.feature_selection import chi2 #feature selector algorithm

import pathlib as path


# MY CLASSES 
from read_database import ReadDatabase as DB
db = DB().getDatabase("/Users/Feti/Desktop/VBOX-001/PythonIAScript/CLASS/Autism_screening dataset.arff");

df = pd.read_csv(db)

#df = df.iloc[:,-0]

df.columns = df.columns.str.replace("/",'_')
df.rename(columns={"austim": "hasAUTISM"}, inplace=True)
df.rename(columns={"jundice": "jaundice"}, inplace=True)
df.rename(columns={"relation": "who_is_talking"}, inplace=True)

df.loc[df.age=='?', 'age'] = 0
df.age = df.age.astype(int)

print("____________________DATASETS__________________")
print('')
print(df)
print("____________________DATASETS SHAPED__________________")
print('')
print(df.shape)
print("____________________DATASETS COLUMNS__________________")
print('')
print(df.columns)
print("____________________DATASETS HEAD__________________")
print('')
print(df.head())
print("____________________DATASETS DESCRIBED__________________")
print('')
print(df.describe())
print("____________________DATASETS CORR__________________")
print('')
print(df.corr())
print('')

df = df.drop_duplicates()
#print(df.shape)
#print(df.isnull().sum())
df = df.dropna()
#print(df.isnull().sum())


gender = {'m': 1, 'f': 0, '?': -1}
df.gender = [gender[item] for item in df.gender]

#print(df.jaundice)
bornwithjaundice = {'yes': 1, 'no': 0}
df.jaundice = [bornwithjaundice[item] for item in df.jaundice]

familymemberwithpdd = {'YES': 1, 'NO': 0}
df.Class_ASD = [familymemberwithpdd[item] for item in df.Class_ASD]
#print(df.Class_ASD)

useda_pp_before = {'yes': 1,'no': 0}
df.used_app_before = [useda_pp_before[item] for item in df.used_app_before]

whos_is_talking = {'Self': 1, 'Parent': 2, "'Health care professional'": 3, 'Relative': 4, 'Others': 5, '?': 6}
df.who_is_talking = [whos_is_talking[item] for item in df.who_is_talking]

#age = {'?': 0}
#df.age = [age[item] for item in df.age]

hasAUTISM = {'yes': 1,'no': 0}
df.hasAUTISM = [hasAUTISM[item] for item in df.hasAUTISM]

target = df.hasAUTISM


X = df[['age', 'gender', 'jaundice', 'Class_ASD', 'used_app_before', 'who_is_talking', 
      'A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 
      'A8_Score', 'A9_Score', 'A10_Score']]

#for t in X.gender:
    #print(t)
    

# Split X and target int X and Y test sets
X_train, X_test, y_train, y_test =  train_test_split(X, target, test_size=0.2, random_state=10)

# Print number of observations X_train, X_test, y_train, y_test
print("____________________X_TRAIN__________________")
print('')
print(X_train.shape)
print(X_train.dtypes)
print('')
print("____________________X_TEST__________________")
print('')
print(X_test.shape)
print('')
print("____________________Y_TRAINS_________________")
print('')
print(y_train.shape)
print(y_train.dtypes)
print('')
print("____________________ Y_TEST__________________")
print('')
print(y_test.shape)
print('')



# 3rd step - load and design the classifiers

from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier,ExtraTreesClassifier,RandomForestClassifier,AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF

def classifiers():
    return [
        SVC(gamma='auto'),
        GaussianProcessClassifier(1.0* RBF(1.0)),
        LinearSVC(),
        SGDClassifier(),
        KNeighborsClassifier(),
        LogisticRegression(solver='lbfgs'),
        LogisticRegressionCV(cv=3),
        BaggingClassifier(),
        ExtraTreesClassifier(n_estimators=300),
        RandomForestClassifier(max_depth=5, n_estimators=300, max_features=1),
        GaussianNB(),
        DecisionTreeClassifier(max_depth=5),
        MLPClassifier(alpha=1,max_iter=1000),
        AdaBoostClassifier(),
        LinearDiscriminantAnalysis(),
        QuadraticDiscriminantAnalysis(),
        OneVsRestClassifier(LinearSVC(random_state=0)),
        GradientBoostingClassifier(),
        SGDClassifier(),
    ]
     





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

# function to do feature selection
def feature_selector(X_train,y_train,X_test,type,i):
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
    print("____________________ THE BEST FEATURE ____________")
    print(bestfeatures)
    fit = bestfeatures.fit(X_train,y_train)
    cols_idxs = fit.get_support(indices=True)
    Xt=X_train.iloc[:,cols_idxs] # extract the best features for training
    Xteste=X_test.iloc[:,cols_idxs] # extract the best features for testing
    return Xt,Xteste


# Function for doing the classification
# the Function receives as inputs:
# models -> machine learning models
# Xt -> training dataframe
# Xteste -> testing dataframe
# yTrain -> training label
def classification(models,Xt,yTrain,Xteste):
    models.fit(Xt, yTrain) # function to train the model
#y_pred means label predictions of the models
    y_pred=models.predict(Xteste) # function for testing the model
    return y_pred


#Load libraries for checking the classification report
from sklearn import metrics
from sklearn.metrics import classification_report
# Function defined for checking the classification performance of the model
# inputs:
# y_test -> the true label
# ypred -> model's label
# outputs: classification report metrics
# accuracy, precision, recall, f1-score
def classification_reports(y_test,ypred):
    report=classification_report(y_test, ypred, output_dict=True)
    accuracy = report['accuracy']
    precision = report['macro avg']['precision']
    recall = report['macro avg']['recall']
    f1 = report['macro avg']['f1-score']
    return [accuracy, precision, recall, f1]


#iterative process to find the e.g. the best 3 features, 4, 5, 6,...,
# until total of features
# classification and models performance checking.

perf_results=pd.DataFrame()
 
from ia_classifiers_model import IAClassifiersModel as classifiers
clfiers = classifiers.classifiers()

for i in range(1,df.shape[1]):
    if(i<=16):
        Xt,Xteste = feature_selector(X_train,y_train,X_test,2,i)
        ypred = classification(clfiers[3],Xt,y_train,Xteste)
        accuracy, precision, recall, f1 = classification_reports(y_test,ypred)
        perf_results[i-3]=[i,accuracy, precision, recall, f1]
    

perf_results=perf_results.T # how to transpose a dataframe collumns <-> rows.
perf_results.columns=['feature','accuracy','precision','recall','f1']
#print the classification of classifier 4 by using the best 3, 4, ..., total of features
print(perf_results)  



