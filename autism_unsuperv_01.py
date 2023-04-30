

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



def normalized_data (df,t):

    if (t==1):
        d=df.copy() # min max normalization
        for each_collum in range(0,df.shape[1]):
            max =df.iloc[:,each_collum].max()
            min =df.iloc[:,each_collum].min()
            d.iloc[:,each_collum]=(d.iloc[:,each_collum]-min)/(max-min)
    elif (t==2):
        d=df.copy() # mean normalization
        for each_collum in range(0,df.shape[1]):
            max =df.iloc[:,each_collum].max()
            min =df.iloc[:,each_collum].min()
            mean =df.iloc[:,each_collum].mean()
            d.iloc[:,each_collum]=(d.iloc[:,each_collum]-mean)/(max-min)
    
    else:
        d=df.copy() # standardization
        for each_collum in range(0,df.shape[1]):
            mean =df.iloc[:,each_collum].mean()
            std =df.iloc[:,each_collum].std()
            d.iloc[:,each_collum]=(d.iloc[:,each_collum]-mean)/(std)

    return d


# 1st step database opening
# MY CLASSES 
from read_database import ReadDatabase as DB
db = DB().getDatabase("/Users/Feti/Desktop/VBOX-001/PythonIAScript/CLASS/Autism_screening dataset.arff");

df = pd.read_csv(db)

df.columns = df.columns.str.replace("/",'_')
df.rename(columns={"austim": "hasAUTISM"}, inplace=True)
df.rename(columns={"jundice": "jaundice"}, inplace=True)
df.rename(columns={"relation": "who_is_talking"}, inplace=True)

df.loc[df.age=='?', 'age'] = 0
df.age = df.age.astype(int)

df = df.drop_duplicates()
#print(df.shape)
#print(df.isnull().sum())
df = df.dropna()



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

#remove target from database
#df=df.iloc[:,:-1]

print(X)

import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(16, 6))

heatmap = sns.heatmap(abs(X.corr()), vmin=0, vmax=1, annot=True)
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':12}, pad=12);
plt.show()

plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(X.corr()[['age']].sort_values(by='age', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Age', fontdict={'fontsize':18}, pad=16);
plt.show()

plt.figure(figsize=(8, 12))
heatmap = sns.heatmap(X.corr()[['gender']].sort_values(by='gender', ascending=False), vmin=-1, vmax=1, annot=True, cmap='BrBG')
heatmap.set_title('Features Correlating with Gender', fontdict={'fontsize':18}, pad=16);
plt.show()



# 3rd step - load and design the classifiers
import matplotlib.pyplot as plt

#The simplest, yet effective clustering algorithm. Needs to be provided with the number of clusters in advance, and assumes that the data is normalized as input (but use a PCA model as preprocessor).
from sklearn.cluster import KMeans #K-Means Clustering

# Can find better looking clusters than KMeans but is not scalable to high number of samples.
from sklearn.cluster import MeanShift #K-Means Clustering

#Clustering algorithm based on message passing between data points.
from sklearn.cluster import AffinityPropagation #K-Means Clustering

#KMeans applied to a projection of the normalized graph Laplacian: finds normalized graph cuts if the affinity matrix is interpreted as an adjacency matrix of a graph.
from sklearn.cluster import SpectralClustering #K-Means Clustering

from sklearn.cluster import AgglomerativeClustering #Hierarchical Clustering

#Can detect irregularly shaped clusters based on density, i.e. sparse regions in the input space are likely to become inter-cluster boundaries. Can also detect outliers (samples that are not part of a cluster).
from sklearn.cluster import DBSCAN #Density-Based Spatial Clustering of Applications with Noise


def classifiers():
    return [
    KMeans(n_clusters=7),
    MeanShift(bandwidth=1),
    AffinityPropagation(),
    SpectralClustering(n_clusters=7),
    AgglomerativeClustering(n_clusters=7),
    DBSCAN(eps=0.3, min_samples=7)
]
    
cla=classifiers()
y_pred=cla[0].fit_predict(X)

from sklearn import metrics
metrics.v_measure_score(y_pred, target)

plt.scatter(X.iloc[:,0], X.iloc[:, 15], c=y_pred)
plt.show()

from sklearn.metrics import confusion_matrix, accuracy_score

print('Accuracy score: ', accuracy_score(target, y_pred))
print(confusion_matrix(target, y_pred))

