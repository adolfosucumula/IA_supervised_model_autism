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

df.columns = df.columns.str.replace("/",'_')
df.rename(columns={"austim": "hasAUTISM"}, inplace=True)
df.rename(columns={"jundice": "jaundice"}, inplace=True)
df.rename(columns={"relation": "who_is_talking"}, inplace=True)


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
print(df.shape)
print(df.isnull().sum())
df = df.dropna()
print(df.isnull().sum())


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
print('')
print("____________________X_TEST__________________")
print('')
print(X_test.shape)
print('')
print("____________________Y_TRAINS_________________")
print('')
print(y_train.shape)
print('')
print("____________________ Y_TEST__________________")
print('')
print(y_test.shape)
print('')


model1 = LinearRegression()
model2 = RandomForestClassifier(n_estimators=500)
model3 = XGBClassifier(n_estimators=500)
model4 = KNeighborsClassifier(n_neighbors=5)
model5 = DecisionTreeClassifier()
model6 = GaussianNB()

# Change the row value from '?' to 0
X_train.loc[X_train.age=='?', 'age'] = 0
X_test.loc[X_test.age=='?', 'age'] = 0

# Training with the models
model1.fit(X_train, y_train)
model2.fit(X_train, y_train)

# Convert the  the datatype of  the column
X_train.age = X_train.age.astype(int)
X_test.age = X_test.age.astype(int)


#Build the model structure
svc = SVC(kernel='linear', C=10.0, random_state=1) #Linear Support vector machine model

#feature selection and classification
for i in range(1,df.shape[1]): #iterative process to find the e.g. the best 3 features, 4, 5, 6,..., until total of features
    if(i <=16):
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
