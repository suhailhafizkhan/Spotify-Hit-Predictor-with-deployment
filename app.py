ju# -*- coding: utf-8 -*-

# All the hyperparameters are chosen in jupyter notebook named Spotify Model building.ipynb 
# this file is only for model development and for deploying. 

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split


df=pd.read_csv("Spotify-data-processed.csv")

df.drop(['Unnamed: 0','track','artist','uri','Year'],axis=1,inplace=True)

sc=StandardScaler()

x=df.iloc[:,0:15].values
y=df.iloc[:,15].values

pyter x_transform=sc.fit_transform(x)

pca_final = PCA(n_components = 10)
new_data = pca_final.fit_transform(x_transform)
principal_x=pd.DataFrame(new_data,columns=["PC-1","PC-2","PC-3","PC-4","PC-5","PC-6","PC-7","PC-8","PC-9","PC-10"])


principal_x

X_train,X_test,Y_train,Y_test=train_test_split(principal_x,y,test_size=0.2,random_state=42)



model=XGBClassifier(learning_rate = 0.05, max_depth = 9, n_estimators = 200)

model.fit(X_train,Y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix,f1_score,roc_curve
acc=accuracy_score(Y_test,y_pred)

import pickle


filename = 'model_pred.pkl'
pickle.dump(model, open(filename, 'wb'))


filename = 'Standard_scaler.pkl'
pickle.dump(sc, open(filename, 'wb'))

filename = 'pca_model.pkl'
pickle.dump(pca_final, open(filename, 'wb'))


f1_score(y_pred,Y_test)

new_df=pd.DataFrame([[0.21,0.62,5,-0.65,1,0.5,0.01,0.5,0.1,0.6,101,173533,4,32.652,9]])


#dat=np.array([0.21,0.62,5,-0.65,1,0.5,0.01,0.5,0.1,0.6,101,173533,4,32.652,9])

#at=dat.reshape(1,-1)
stand=sc.transform(new_df)

#pca_final1 = PCA(n_components = 10)
#cols=df.columns

pica=pca_final.transform(stand)

principalsx=pd.DataFrame(pica,columns=["PC-1","PC-2","PC-3","PC-4","PC-5","PC-6","PC-7","PC-8","PC-9","PC-10"])


preeee=model.predict(principalsx)
if preeee[0] == 0:
    print("flop")
else:
    print("hit")