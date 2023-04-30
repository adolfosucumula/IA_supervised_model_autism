

import pandas as pd
import numpy as np #Tool box to work with array

from read_database import ReadDatabase as DB
from normalizer_IA_model import NormalizerIAModel as Normalizer
from feature_selector_model import FeatureSelectorModel as FSM

#from read_database import getDatabase

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import BaggingClassifier, ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.gaussian_process.kernels import RBF

from sklearn.metrics import classification_report # Get reports
from sklearn.feature_selection  import SelectKBest # The way to perfome the feature selection
from sklearn.feature_selection import chi2, mutual_info_classif, f_regression, f_classif, mutual_info_regression, SelectPercentile, SelectFdr, SelectFpr, SelectFwe, GenericUnivariateSelect


##


##

classifiers = [
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


#v = int(input("Type the normalization: "))

db = DB().getDatabase();
norm = Normalizer()
fs = FSM();

df = pd.read_csv(db, ";", na_values=['--','n/a', 'nada'])

# split the database in target and features
target=df.iloc[:,-1]
df=df.iloc[:,:-1]
print(df)
# 2nd step - features normalization
d_n = norm.normalize_data(df, 1)

# split data for training and testing your model
X_train, X_test, y_train, y_test = train_test_split(d_n, target, train_size = 0.75) # 75% of data goes for training and 25% goes for testing


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

for i in range(3,d_n.shape[1]):
    Xt,Xteste = fs.feature_selector(X_train,y_train,X_test,2,i)
    ypred = classification(classifiers[3],Xt,y_train,Xteste)
    accuracy, precision, recall, f1 = classification_reports(y_test,ypred)
    perf_results[i-3]=[i,accuracy, precision, recall, f1]

perf_results=perf_results.T # how to transpose a dataframe collumns <-> rows.
perf_results.columns=['feature','accuracy','precision','recall','f1']
#print the classification of classifier 4 by using the best 3, 4, ..., total of features

#print(perf_results)   

    