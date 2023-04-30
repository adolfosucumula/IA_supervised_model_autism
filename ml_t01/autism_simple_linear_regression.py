
# MY CLASSES 

from read_database import ReadDatabase as DB
#from normalizer_IA_model import NormalizerIAModel as Normalizer
#from feature_selector_model import FeatureSelectorModel as FSM

######################

import matplotlib.pyplot as plt
import pandas as pd
import pylab as pl
import numpy as np


db = DB().getDatabase("/Users/Feti/Desktop/VBOX-001/PythonIAScript/CLASS/Autism_screening dataset.arff");

df = pd.read_csv(db)
df = pd.DataFrame(df)

df.loc[df.austim=='no', 'austim'] = 0
df.loc[df.austim=='yes', 'austim'] = 1
df.austim = df.austim.astype(int)

# take a look at the dataset
df.head();

# summarize the data
df.describe()
print(df.dtypes)
cdf = df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'result', 'austim']]
cdf.head(9)

wiz = df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'result', 'austim']]
#wiz.hist()
#plt.show()

plt.scatter(cdf.A3_Score, cdf.austim, color ='blue')
plt.xlabel("A2 SCORE")
plt.ylabel("AUSTIM")
plt.show()


