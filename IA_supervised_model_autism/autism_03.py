#for numerical coputing
import numpy as np

#for dataforms
import pandas as pd

#import warnings
import warnings as wng
wng.filterwarnings("ignore")

# To split train and test sets
from sklearn.model_selection import train_test_split

from sklearn.metrics import accuracy_score

# Machine Learn Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


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

model3.fit(X_train, y_train)
model4.fit(X_train, y_train)
model5.fit(X_train, y_train)
model6.fit(X_train, y_train)


# Predict test set results

Y_pred_1 = model1.predict(X_test)
Y_pred_2 = model2.predict(X_test)
Y_pred_3 = model3.predict(X_test)
Y_pred_4 = model4.predict(X_test)
Y_pred_5 = model5.predict(X_test)
Y_pred_6 = model6.predict(X_test)

print("____________________ Y_PREDICTED 1 __________________")
print('')
print(Y_pred_1)
print('')

print("____________________ Y_PREDICTED 2 __________________")
print('')
print(Y_pred_2)
print('')

print("____________________ Y_PREDICTED 3 __________________")
print('')
print(Y_pred_3)
print('')

print("____________________ Y_PREDICTED 4 __________________")
print('')
print(Y_pred_4)
print('')

print("____________________ Y_PREDICTED 5 __________________")
print('')
print(Y_pred_5)
print('')

print("____________________ Y_PREDICTED 6 __________________")
print('')
print(Y_pred_6)
print('')

#acc = accuracy_score(y_test, Y_pred_1) # Get the accuracy from testing data
#print("Accuracy of Linear Regression: {1.2f}%", format(acc*100) )

acc = accuracy_score(y_test, Y_pred_2) # Get the accuracy from testing data
print("Accuracy of RandomForestClassifier: {1.2f}%", format(acc*100) )

acc = accuracy_score(y_test, Y_pred_3) # Get the accuracy from testing data
print("Accuracy of XGSClassifier: {1.2f}%", format(acc*100) )

acc = accuracy_score(y_test, Y_pred_4) # Get the accuracy from testing data
print("Accuracy of KNeighborsClassifier: {1.2f}%", format(acc*100) )

acc = accuracy_score(y_test, Y_pred_5) # Get the accuracy from testing data
print("Accuracy of Decision Tree Classifier: {1.2f}%", format(acc*100) )

acc = accuracy_score(y_test, Y_pred_6) # Get the accuracy from testing data
print("Accuracy of GaussianNB: {1.2f}%", format(acc*100) )












#import joblib 
#joblib.dump(model1, 'ASD_final.pk1')
#final_model = joblib.load('ASD_final.pk1')

#pred = final_model.predict(X_test)
#acc = accuracy_score(y_test, pred)
#print('Final Model Accuracy: {1.2f}%', format(acc*100))