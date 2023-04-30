
import pandas as pd #toolbox to work with dataframes
import numpy as np #toolbox to work with narrays
import matplotlib.pyplot as plt #toolbox to do plots
from sklearn.svm import SVC #load the support vector machine model functions
from sklearn.model_selection import train_test_split #load the function to split train and test sets
from sklearn import metrics # get the report
from sklearn.metrics import classification_report # get the report
from sklearn import preprocessing # normalize the features
from sklearn.preprocessing import MinMaxScaler # normalize the features
from sklearn.feature_selection import SelectKBest #load the feature selector model  
from sklearn.feature_selection import chi2 #feature selector algorithm

#the objective is  to predict the quality of the wine 

df = pd.read_csv('winequality-white.csv',sep = ';') #open the database


Database_n = pd.DataFrame(preprocessing.normalize(df.iloc[:,:-1])) #All dataset normalization except the last collumn related with target
target = df.iloc[:,df.shape[1]-1] #store the target in a variable

#split the data in training and testing sets
X_train, X_test, y_train, y_test = train_test_split(Database_n, target, train_size = 0.75)

#Build the model structure
svc = SVC(kernel='linear', C=10.0, random_state=1) #Linear Support vector machine model

#feature selection and classification
for i in range(1,Database_n.shape[1]): #iterative process to find the e.g. the best 3 features, 4, 5, 6,..., until total of features
    bestfeatures = SelectKBest(score_func=chi2, k=i)
    fit = bestfeatures.fit(X_train,y_train)# always select the best features within the training set!
    cols_idxs = fit.get_support(indices=True)
    Xt=X_train.iloc[:,cols_idxs] #get the best feature select depending of the iteration
    Xteste=X_test.iloc[:,cols_idxs] #you need to test with same features that have been used for training
    svc.fit(Xt, y_train) # function to train the model
    y_pred=svc.predict(Xteste) # function for testing the model
    report=classification_report(y_test, y_pred, output_dict=True) # function to get the classification report
    accuracy = report['accuracy'] #get the accuracy
    print(accuracy) # print the accuracy
