


import pathlib as path

# MY CLASSES 

from read_database import ReadDatabase as DB
#from normalizer_IA_model import NormalizerIAModel as Normalizer
#from feature_selector_model import FeatureSelectorModel as FSM

######################

#import matplotlib as plt

# import a function to load data in the arff format, which is the format of the children's autism data from UCI
from scipy.io import arff
# import numpy, the main python library for creating and manipulating matrices, which are usually called arrays in numpy
import numpy as np
# import pandas, a handy library for handling tabular data
import pandas as pd
# import the models we're going to try from scikit-learn, a powerful general purpose machine learning library
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

#pip install scikit-neuralnetwork 
#from sknn.mlp import Classifier, Layer
# import functions for evaluating and comparing models
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV 
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
# import plotting libraries

import matplotlib.pyplot as plt 
import seaborn as sns

#file = path.Path("winequality-white.csv");
#db = pd.read_csv(file);

#target = db.iloc[:,-1]

#training model

#model =LinearRegression()
#find the optmized line
#print(model.fit(target))


# define a "seed" to be used for any process that involves random number generation, including numpy's
# this is to ensure that we can reproduce the results of any experiment, even if it involves "random" processes
random_seed = 1
np.random.seed(random_seed)
# load the data

#read database
#Reading the data
db = DB().getDatabase("/Users/Feti/Desktop/VBOX-001/PythonIAScript/CLASS/Autism_screening dataset.arff");

#file = "/Users/Feti/Desktop/VBOX-001/PythonIAScript/CLASS/Autism_screening dataset.arff"
#data = arff.loadarff(file)

# convert the data into a pandas dataframe (a kind of table) for easier manipulation
data = pd.read_csv(db, ",");

df = pd.DataFrame(data)

#print(df)
# shuffle the order of the patients, just in case the author of the dataset ordered them by diagnosis or something
df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)

# variables to store the dimensions of the data
data_dimension = df.shape
case_count = df.shape[0]
column_count = df.shape[1]

# make sure all columns from the dataframe are displayed
pd.set_option('display.max_columns', column_count)

# display the first five rows of the dataframe
df.head()

# look at summary statistics of the overall dataset
df.describe(include='all')

# check whether there are any missing values. there are some in age: we'll deal with this later during preprocessing
df.isnull().sum()

# store the name of the outcome column in a variable
label_column = 'Class/ASD'

# filter the dataset so only positive cases are left
positive_df = df.loc[df[label_column]==b'YES']

# look at summary statistics for just positive cases
positive_df.describe(include='all')

# filter the dataset so only negative cases are left
negative_df = df.loc[df[label_column]=='NO']

# look at summary statistics for just negative cases
negative_df.describe(include='all')

# create a dataframe with only the AQ-10 and "result" columns
aq10_columns = ['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score',
'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score']

aq10_and_result_columns = aq10_columns + ['result']
aq10_and_result_df = df[aq10_and_result_columns]
aq10_and_result_df.head()

# using a function from the statsmodels package, determine whether any column can be perfectly predicted from the others
from statsmodels.stats.outliers_influence import variance_inflation_factor

# convert the aq-10 dataframe into a numpy array of the float datatype for analysis by this function
aq10_and_result_df = aq10_and_result_df.astype(float)
aq10_and_result_matrix = aq10_and_result_df.values

# run the function
vif = [variance_inflation_factor(aq10_and_result_matrix, i) for i in range(aq10_and_result_matrix.shape[1])]
# display the results. a value of "inf" means that a given column can be perfectly predicted from the others

# check how many cases there are where the "result" column isn't the sum of the AQ-10 columns
# create a dataframe that only contains the AQ-10 columns
aq10_df = aq10_and_result_df[aq10_columns]

# create an array whose values are the sums of the AQ-10 columns for each case
aq10_sum = np.sum(aq10_df, axis=1)!=aq10_and_result_df.result

# create an array whose values indicate whether or not the AQ-10 sum for a given case is different from the "result" value for that case
sum_different_from_result = np.sum(aq10_df, axis=1)!=aq10_and_result_df.result

# count the number of cases where the result isn't just the sum of the AQ-10 answers 
np.sum(sum_different_from_result)

# create a dataframe with only the categorical feature columns we want to use 
categorical_features = ['gender', 'jundice', 'austim', 'contry_of_res', 'ethnicity'] 
categorical_feature_df = df[categorical_features]

# display the dimensions of the dataframe
print("Original dimensions: " + str(categorical_feature_df.shape))

# use a built-in pandas function to convert non-binary categorical variables into a set of binary "dummy" variables.
# this is necessary in order to feed this data to scikit-learn models later.

categorical_feature_df = pd.get_dummies(categorical_feature_df, drop_first=True)
# convert from dataframe to numpy array
X_categorical = categorical_feature_df.values
# display dimensions of the new array; notice how many more columns there are due to the dummy variable transformation
print("New dimensions: " + str(X_categorical.shape))

#df.columns = db.columns.str.replace("/", "_")

# create a dataframe with only the numerical features we want to use, in this case just age 
numerical_features = ['age']
numerical_feature_df = df[numerical_features]
numerical_feature_df = pd.DataFrame(numerical_feature_df)
numerical_feature_df.loc[numerical_feature_df.age=='?', 'age']=0
numerical_feature_df.age = numerical_feature_df.age.astype(int)

# calculate the average age
mean_age = np.mean(numerical_feature_df['age'])

# fill in the 4 missing values of age we discovered earlier with the average, a common way of handling missing values
numerical_feature_df = numerical_feature_df.fillna(mean_age)
X_numerical = numerical_feature_df.values

# check for missing values again to make sure we fixed them 
numerical_feature_df.isnull().sum()

# create array of all features to be used for training
X = np.append(X_numerical, X_categorical, axis=1)

# create binary array for the label column
y = df[label_column].values==b'YES'

# split the data 80/20 into a training and test set

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.2, random_state=random_seed)

print("TRAIN DATASETS")
print('')
print(X_train)
print(y_train)
print('')
print("TEST DATASETS")
print('')
print(X_test)
print(y_test)
