
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV


df=pd.read_csv('D:/Datasets/Hackathon/train_dataset.csv')
df_test=pd.read_csv('D:/Datasets/Hackathon/test_dataset.csv')

pd.set_option('display.max_columns', 20)

df 
df.head() 
df.Drug 
df[6797:6798]
df.Age=round(df.Age/365,1) 
df_test.Age=round(df_test.Age/365,1)
df.Age 

df.isnull().sum() 
df.Age.mean() 
df.describe()
df.Age 
df.Drug=df.Drug.fillna(method='bfill')
df.Drug.isnull().sum()
df.Ascites=df.Ascites.fillna(method='ffill')
df.Ascites 

### Removing the null values on train set

df['Hepatomegaly']=df['Hepatomegaly'].fillna(method='ffill')
df['Spiders']=df['Spiders'].fillna(method='ffill')
df['Edema'].isnull().sum()    
df.Cholesterol=df.Cholesterol.fillna(df.Cholesterol.mean())
df.Prothrombin=df.Prothrombin.fillna(df.Prothrombin.mean())
df.Platelets=df.Platelets.fillna(df.Platelets.mean())
df.Tryglicerides=df.Tryglicerides.fillna(df.Tryglicerides.mean())
df.SGOT=df.SGOT.fillna(df.SGOT.mean())
df.Copper=df.Copper.fillna(df.Copper.mean())
df.Alk_Phos=df.Alk_Phos.fillna(df.Alk_Phos.mean())
df.Spiders=df.Spiders.fillna(method='ffill')
df.Hepatomegaly=df.Hepatomegaly.fillna(method='ffill')

### Label encoding the columns

le=LabelEncoder()

df.Status=le.fit_transform(df.Status)
df.Drug=le.fit_transform(df.Drug)
df.Sex=le.fit_transform(df.Sex)
df.Ascites=le.fit_transform(df.Ascites)
df.Hepatomegaly=le.fit_transform(df.Hepatomegaly)
df.Spiders=le.fit_transform(df.Spiders)
df.Edema=le.fit_transform(df.Edema)



### Visualizing the data

print(df.corr())
sns.heatmap(df.corr())
plt.hist(df.Bilirubin, bins=50)##right skewed data
plt.hist(df.Cholesterol, bins=50, label='Cholestrol',)

### train test split

y=df.iloc[:,-1]
X=df.iloc[:,2:19]
y 

X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=30, train_size=70, random_state=1)
X_train  
y_train 

sns.heatmap(X)


### Random Forest Classifier also using Grid Search CV for hyperparameter tuning
RF=RandomForestClassifier(max_depth=4, max_features='auto',n_estimators=350, n_jobs=4)
parameters={'n_estimators':(120,180,250,280,350), 'criterion':('gini', 'entropy'), 'max_depth':(4,8,10,12), 'max_features':('auto','sqrt','log2'), 'n_jobs':(0,2,4,6,8)}
clf=GridSearchCV(RF,parameters)

clf.fit(X_train,y_train)
clf.cv_results_.keys()
clf.best_params_
RF.fit(X_train,y_train)
y_pred=RF.predict(X_test)
y_pred 

score=accuracy_score(y_test,y_pred)
score 
cmatrix=confusion_matrix(y_test,y_pred) 
cmatrix 
f1=f1_score(y_test,y_pred, average='weighted')
f1 

###preparing the test file

df_test.isnull().sum() 
df_test 

### Removing the null values on test set

df_test.Drug=df_test.Drug.fillna(method='bfill')
df_test.Drug.isnull().sum()
df_test.Ascites=df_test.Ascites.fillna(method='ffill')
df_test.Ascites 


df_test['Hepatomegaly']=df_test['Hepatomegaly'].fillna(method='ffill')
df_test['Spiders']=df_test['Spiders'].fillna(method='ffill')
df_test['Edema'].isnull().sum()    
df_test.Cholesterol=df_test.Cholesterol.fillna(df_test.Cholesterol.mean())
df_test.Prothrombin=df_test.Prothrombin.fillna(df_test.Prothrombin.mean())
df_test.Platelets=df_test.Platelets.fillna(df_test.Platelets.mean())
df_test.Tryglicerides=df_test.Tryglicerides.fillna(df_test.Tryglicerides.mean())
df_test.SGOT=df_test.SGOT.fillna(df_test.SGOT.mean())
df_test.Copper=df_test.Copper.fillna(df_test.Copper.mean())
df_test.Alk_Phos=df_test.Alk_Phos.fillna(df_test.Alk_Phos.mean())
df_test.Hepatomegaly=df_test.Hepatomegaly.fillna(method='bfill')

### Label encoding the columns

le=LabelEncoder()

df_test.Status=le.fit_transform(df_test.Status)
df_test.Drug=le.fit_transform(df_test.Drug)
df_test.Sex=le.fit_transform(df_test.Sex)
df_test.Ascites=le.fit_transform(df_test.Ascites)
df_test.Hepatomegaly=le.fit_transform(df_test.Hepatomegaly)
df_test.Spiders=le.fit_transform(df_test.Spiders)
df_test.Edema=le.fit_transform(df_test.Edema)


XR_test=df_test.iloc[:,2:]
XR_test 
y_pred_final=RF.predict(XR_test)

y_pred_final 
result=pd.DataFrame(y_pred_final)

result.rename(columns= {0:'Stage'}, inplace=True) 
result.to_csv('D:/Datasets/Hackathon/result.csv',header=True,index=None)

