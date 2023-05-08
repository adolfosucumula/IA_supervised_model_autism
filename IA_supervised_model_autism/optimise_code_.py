import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV

df = pd.read_csv('winequality-white.csv',';')

for i in range(0,df.shape[0]):
    if (df.iloc[i,-1]>5):
        df.iloc[i-1] = 1
    else:
        df.iloc[i,-1]=0
        
features = df.iloc[:,:-1]
target = df.iloc[:,-1]

rf = RandomForestClassifier()

#Parametros a serem optimizados
grid_space = {'max_depth':[3,5,10,None],
              'n_estimators':[10,100,200,500,1000],
              'min_samples_split':[1,2,3],
              'min_samples_leaf':[1,2,3]}

grid = GridSearchCV(rf, param_grid=grid_space, cv=3, scoring='accuracy')

model_grid = grid.fit(features, target)

print('Best grid searc hyperparameters are: ' + str(model_grid.best_params_))
print('Best grid search score is: ' + str(model_grid.best_score_))