#Imports
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt  

from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import classification_report
from sklearn.model_selection import RandomizedSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
import seaborn as sns
import pickle

#Load
df=pd.read_csv('https://raw.githubusercontent.com/4GeeksAcademy/random-forest-project-tutorial/main/titanic_train.csv')

#Cleaning data
X = df.drop('Survived', axis=1)
y = df['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)
df_train = pd.concat([X_train, y_train], axis=1)

drop_cols = ['PassengerId','Cabin', 'Ticket', 'Name']
df_train = df_train.drop(drop_cols, axis = 1)

X = df_train.drop('Survived', axis=1)
y = df_train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1234)
df_train = pd.concat([X_train, y_train], axis=1)

#Customizing 
imputer_mean = SimpleImputer(strategy='mean', missing_values=np.nan)
imputer_mean = imputer_mean.fit(X_train[['Age']])
X_train['Age'] = imputer_mean.transform(X_train[['Age']])


imputer_mode = SimpleImputer(strategy='most_frequent', missing_values=np.nan)
imputer_mode = imputer_mode.fit(X_train[['Embarked']])
X_train['Embarked'] = imputer_mode.transform(X_train[['Embarked']])

X_test['Age'] = imputer_mean.transform(X_test[['Age']])

X_test['Embarked'] = imputer_mode.transform(X_test[['Embarked']])

X_train[['Sex','Embarked']]=X_train[['Sex','Embarked']].astype('category')
X_test[['Sex','Embarked']]=X_test[['Sex','Embarked']].astype('category')


X_train['Sex']=X_train['Sex'].cat.codes
X_train['Embarked']=X_train['Embarked'].cat.codes

X_test['Sex']=X_test['Sex'].cat.codes
X_test['Embarked']=X_test['Embarked'].cat.codes

#Random Forest
rfc = RandomForestClassifier(random_state=1107)

rfc.fit(X_train, y_train)

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
#Criterio
criterion=['gini','entropy']

# Create the random grid
random_grid = {'n_estimators': n_estimators,
#'max_features': max_features, # Son muy pocas variables por lo cual no vale la pena aplicarlo
'max_depth': max_depth,
'min_samples_split': min_samples_split,
'min_samples_leaf': min_samples_leaf,
'bootstrap': bootstrap,
'criterion':criterion}

rfc3=RandomForestClassifier(random_state=1107)

grid_random=RandomizedSearchCV(estimator=rfc3,n_iter=100,cv=5,random_state=1107,param_distributions=random_grid)

grid_random.fit(X_train,y_train)

best_param = grid_random.best_params_
best_model = RandomForestClassifier(**best_param)
best_model

model_cv_2 = grid_random.best_estimator_

y_pred_cv_2 = model_cv_2.predict(X_test)

cm = confusion_matrix(y_test, y_pred_cv_2, labels=grid_random.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
display_labels=grid_random.classes_)
disp.plot()

plt.show()

filename = '../models/final_model.sav'
pickle.dump(model_cv_2, open(filename, 'wb'))


#XGBOOST
xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')

xgb.fit(X_train, y_train)
print('Test_ac:',accuracy_score(y_test,y_xgb_pred))

xgb_2 = XGBClassifier()
parameters = {
     "eta"    : [0.05, 0.10, 0.15, 0.20, 0.25, 0.30 ] ,
     "max_depth"        : [ 3, 4, 5, 6, 8, 10, 12, 15],
     "min_child_weight" : [ 1, 3, 5, 7 ],
     "gamma"            : [ 0.0, 0.1, 0.2 , 0.3, 0.4 ],
     "colsample_bytree" : [ 0.3, 0.4, 0.5 , 0.7 ]
     }

grid_xgb = RandomizedSearchCV(xgb_2,
                    parameters, n_jobs=4,
                    scoring="neg_log_loss",
                    cv=3)

grid_xgb.fit(X_train, y_train)

xgb_2 = grid_xgb.best_estimator_

y_pred_xgb_2 = xgb_2.predict(X_test)

print('Test_ac:',accuracy_score(y_test, y_pred_xgb_2))

filename = '../models/xgboost_model.sav'
pickle.dump(grid_xgb, open(filename, 'wb'))