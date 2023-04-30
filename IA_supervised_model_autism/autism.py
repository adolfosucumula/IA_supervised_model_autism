
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pathlib as path

from sklearn.metrics import classification_report # get the report
from sklearn import preprocessing # normalize the features
from sklearn.model_selection import train_test_split #load the function to split train and test sets
from sklearn.svm import SVC #load the support vector machine model functions
from sklearn.feature_selection import SelectKBest #load the feature selector model  
from sklearn.feature_selection import chi2 #feature selector algorithm

# MY CLASSES 

from read_database import ReadDatabase as DB
from normalizer_IA_model import NormalizerIAModel as Normalizer
from feature_selector_model import FeatureSelectorModel as FSM

######################

#read database
#Reading the data
db = DB().getDatabase("/Users/Feti/Desktop/VBOX-001/PythonIAScript/CLASS/Autism_screening dataset.arff");
norm = Normalizer()
fs = FSM();

#df = pd.read_csv(db, ",", na_values=['--', 'NO', 'YES', 'no', '?', 'f']);

#df = pd.DataFrame(df)



db = pd.read_csv(db, ",", na_values=['--', '?', 'f']);

df = pd.DataFrame(db)
#df = df.drop(columns=['relation', 'result', 'contry_of_res', 'used_app_before', 
                      #'age_desc','gender', 'jundice', 'austim','ethnicity','Class/ASD'])
#df.head()


df.columns = db.columns.str.replace("/", "_") #Change a type of character: https://raccoon.ninja/pt/dev-pt/renomeando-colunas-do-dataframe-python-pandas-jupyter-dev-data/

df.loc[df.Class_ASD=='NO','Class_ASD']=0 #Change the value of a column from 'NO' to 0
df.loc[df.Class_ASD=='YES','Class_ASD']=1 #Change the value of a column from 'YES' to 1
df.loc[df.used_app_before == 'no', 'used_app_before'] = 0
df.loc[df.used_app_before == 'yes', 'used_app_before'] = 1

# convert data type of grade column https://acervolima.com/alterar-o-tipo-de-dados-de-uma-coluna-ou-serie-pandas/
# into integer
df.Class_ASD = df.Class_ASD.astype(int) 
df.used_app_before = df.used_app_before.astype(int)

features= df.iloc[:,:11]# select the features from 0 to 11
features2= df.iloc[:,13:19]#select the features from 13 till 19
features3=df.iloc[:,20:]#select the features from 20 till last feature
features_scores=df.iloc[:,:11]
features_f=pd.concat ([features,features2,features3],axis=1)#add the two features together
target1=df.iloc[:,14]
#print(features_f)#print f
#print(dn)



features_f = features_f.drop(columns=['result', 'contry_of_res', 
                      'age_desc', 'jundice', 'austim'])
features_f.head()

features_f = features_f[np.isfinite(features_f).all(1)] #remove rows with any values that are not finite: https://www.statology.org/input-contains-nan-infinity-or-value-too-large-for-dtype/

# 2nd step - features normalization
features_f = norm.normalize_data(features_f, 1)
#features_f = pd.DataFrame(preprocessing.normalize(features_f.iloc[:,:-1])) #All dataset normalization except the last collumn related with target
#target2=df.iloc[:,14]
target3 = features_f.iloc[:,features_f.shape[1]-1] #store the target in a variable

#print(features_f)
#print(features_f.dtypes)

#target = df.iloc[:,df.shape[1]-1] #store the target in a variable


#split the data in training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features_f, target3, train_size = 0.75)

#print(X_train)
#print(target3)

#Build the model structure
svc = SVC(kernel='linear', C=10.0, random_state=1) #Linear Support vector machine model

#feature selection and classification
for i in range(1,db.shape[1]): #iterative process to find the e.g. the best 3 features, 4, 5, 6,..., until total of features
    
    bestfeatures = SelectKBest(score_func=chi2, k=13)
    fit = bestfeatures.fit(X_train,y_train)# always select the best features within the training set!
    cols_idxs = fit.get_support(indices=True)
    Xt=X_train.iloc[:,cols_idxs] #get the best feature select depending of the iteration
    Xteste=X_test.iloc[:,cols_idxs] #you need to test with same features that have been used for training
    svc.fit(Xt, y_train) # function to train the model
    y_pred=svc.predict(Xteste) # function for testing the model
    report=classification_report(y_test, y_pred, output_dict=True) # function to get the classification report
    accuracy = report['accuracy'] #get the accuracy
    print(accuracy) # print the accuracy
