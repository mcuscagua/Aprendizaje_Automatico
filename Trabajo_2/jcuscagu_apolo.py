# -*- coding: utf-8 -*-
"""
Entrenamiento de Perceptron Multicapa

@author: Juan Mauricio Cuscagua Lopez
"""

import numpy as np
import pandas as pd
import itertools as it
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, learning_curve
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_curve, precision_score, accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pylab as plt

import dropbox

import plotly.graph_objects as go

standarscaler = StandardScaler()

from warnings import filterwarnings
filterwarnings('ignore')


TKN = 'lu7SdhMk0DkAAAAAAAAHEqMhPlhoMgWrehCDWA9zZLi6xwxQf78qvD18-H145tuI'
path = '/Aprendizaje_Automatico/'


def to_dropbox(dataframe, path, token):

    dbx = dropbox.Dropbox(token)

    df_string = dataframe.to_csv(index=False)
    db_bytes = bytes(df_string, 'utf8')
    dbx.files_upload(
        f=db_bytes,
        path=path,
        mode=dropbox.files.WriteMode.overwrite
    )
    

Data = pd.read_csv('https://raw.githubusercontent.com/mcuscagua/Aprendizaje_Automatico/master/Trabajo_2/Data.csv')
Data = Data.set_index('Date')
Y = Data['Class']
X = Data.drop('Class', axis = 1)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

X_train_Stand = standarscaler.fit_transform(X_train)
X_test_Stand = standarscaler.transform(X_test)


Neurons = [x+1 for x in range(10)]
all_combinations = []
for i in range(len(Neurons)):
    all_combinations += [p for p in it.product(Neurons, repeat=i+1)]
    
TSZ_LC_V02 = np.ndarray((len(all_combinations),5))
TrSCr_LC_V02 = np.ndarray((len(all_combinations),5))
TeSCr_LC_V02 = np.ndarray((len(all_combinations),5))
AUC_v02 = np.zeros(len(all_combinations))
Accuracy_v02 = np.zeros(len(all_combinations))

TSZ_LC_V05 = np.ndarray((len(all_combinations),5))
TrSCr_LC_V05 = np.ndarray((len(all_combinations),5))
TeSCr_LC_V05 = np.ndarray((len(all_combinations),5))
AUC_v05 = np.zeros(len(all_combinations))
Accuracy_v05 = np.zeros(len(all_combinations))

TSZ_LC_V09 = np.ndarray((len(all_combinations),5))
TrSCr_LC_V09 = np.ndarray((len(all_combinations),5))
TeSCr_LC_V09 = np.ndarray((len(all_combinations),5))
AUC_v09 = np.zeros(len(all_combinations))
Accuracy_v09 = np.zeros(len(all_combinations))


for i in range(len(all_combinations)):
    #--------- LR = 0.2 ---------#
    MLPC = MLPClassifier(hidden_layer_sizes=all_combinations[i],
                             learning_rate = 'constant',
                             learning_rate_init = 0.2,
                             max_iter = 50,
                             tol = 1e-2,
                             random_state=0)

    MLPP = MLPClassifier(hidden_layer_sizes=all_combinations[i],
                             learning_rate = 'constant',
                             learning_rate_init = 0.2,
                             max_iter = 50,
                             tol = 1e-2,
                             random_state=0)

    MLPC.fit(X_train_Stand, Y_train)
    y_score = MLPC.predict_proba(X_test_Stand)
    y_pred = MLPC.predict(X_test_Stand)

    train_sizes, train_scores, test_scores = learning_curve(MLPP, X, Y)
    
    TSZ_LC_V02[i,:] = train_sizes
    TrSCr_LC_V02[i,:] = np.median(train_scores, axis=1)
    TeSCr_LC_V02[i,:] = np.median(test_scores, axis=1)
    AUC_v02[i] = roc_auc_score(Y_test, y_pred)
    Accuracy_v02[i] = accuracy_score(Y_test, y_pred)
    
    #--------- LR = 0.5 ---------#
    
    MLPC = MLPClassifier(hidden_layer_sizes=all_combinations[i],
                             learning_rate = 'constant',
                             learning_rate_init = 0.5,
                             max_iter = 50,
                             tol = 1e-2,
                             random_state=0)

    MLPP = MLPClassifier(hidden_layer_sizes=all_combinations[i],
                             learning_rate = 'constant',
                             learning_rate_init = 0.5,
                             max_iter = 50,
                             tol = 1e-2,
                             random_state=0)

    MLPC.fit(X_train_Stand, Y_train)
    y_score = MLPC.predict_proba(X_test_Stand)
    y_pred = MLPC.predict(X_test_Stand)

    train_sizes, train_scores, test_scores = learning_curve(MLPP, X, Y)
    
    TSZ_LC_V05[i,:] = train_sizes
    TrSCr_LC_V05[i,:] = np.median(train_scores, axis=1)
    TeSCr_LC_V05[i,:] = np.median(test_scores, axis=1)
    AUC_v05[i] = roc_auc_score(Y_test, y_pred)
    Accuracy_v05[i] = accuracy_score(Y_test, y_pred)
    
    #--------- LR = 0.9 ---------#
    
    MLPC = MLPClassifier(hidden_layer_sizes=all_combinations[i],
                             learning_rate = 'constant',
                             learning_rate_init = 0.9,
                             max_iter = 50,
                             tol = 1e-2,
                             random_state=0)

    MLPP = MLPClassifier(hidden_layer_sizes=all_combinations[i],
                             learning_rate = 'constant',
                             learning_rate_init = 0.9,
                             max_iter = 50,
                             tol = 1e-2,
                             random_state=0)

    MLPC.fit(X_train_Stand, Y_train)
    y_score = MLPC.predict_proba(X_test_Stand)
    y_pred = MLPC.predict(X_test_Stand)

    train_sizes, train_scores, test_scores = learning_curve(MLPP, X, Y)
    
    TSZ_LC_V09[i,:] = train_sizes
    TrSCr_LC_V09[i,:] = np.median(train_scores, axis=1)
    TeSCr_LC_V09[i,:] = np.median(test_scores, axis=1)
    AUC_v09[i] = roc_auc_score(Y_test, y_pred)
    Accuracy_v09[i] = accuracy_score(Y_test, y_pred)
    

TSZ_LC_V02_DF = pd.DataFrame(TSZ_LC_V02[0,:])

TRSCR_DF_02 = pd.DataFrame(TrSCr_LC_V02)
TRSCR_DF_05 = pd.DataFrame(TrSCr_LC_V05)
TRSCR_DF_09 = pd.DataFrame(TrSCr_LC_V09)

TESCR_DF_02 = pd.DataFrame(TeSCr_LC_V09)
TESCR_DF_05 = pd.DataFrame(TeSCr_LC_V05)
TESCR_DF_09 = pd.DataFrame(TeSCr_LC_V09)

configurations = pd.DataFrame(all_combinations)

AUC_MAT = pd.DataFrame([AUC_v02, AUC_v05, AUC_v09]).transpose()
ACC_MAT = pd.DataFrame([Accuracy_v02, Accuracy_v05, Accuracy_v09]).transpose()


Splitter = np.arange(11111111110, step = 25000000)
Rows = AUC_MAT.shape[0]

for i in range(len(Splitter)-1):
    if i == 0:

        to_dropbox(TRSCR_DF_02.iloc[Splitter[i]:min(Splitter[i+1], Rows),],
                   path + 'Train_Score_LC/TraninScoreLR02_'+str(i)+'.csv',
                   TKN)
        to_dropbox(TRSCR_DF_05.iloc[Splitter[i]:min(Splitter[i+1], Rows),],
                   path + 'Train_Score_LC/TraninScoreLR05_'+str(i)+'.csv',
                   TKN)
        to_dropbox(TRSCR_DF_09.iloc[Splitter[i]:min(Splitter[i+1], Rows),],
                   path + 'Train_Score_LC/TraninScoreLR09_'+str(i)+'.csv',
                   TKN)
        
        to_dropbox(TESCR_DF_02.iloc[Splitter[i]:min(Splitter[i+1], Rows),],
                   path + 'Test_Score_LC/TraninScoreLR02_'+str(i)+'.csv',
                   TKN)
        to_dropbox(TESCR_DF_05.iloc[Splitter[i]:min(Splitter[i+1], Rows),],
                   path + 'Test_Score_LC/TraninScoreLR05_'+str(i)+'.csv',
                   TKN)
        to_dropbox(TESCR_DF_09.iloc[Splitter[i]:min(Splitter[i+1], Rows),],
                   path + 'Test_Score_LC/TraninScoreLR09_'+str(i)+'.csv',
                   TKN)
        
        to_dropbox(configurations.iloc[Splitter[i]:min(Splitter[i+1], Rows),],
                   path + 'Configurations/Arquitectura_'+str(i)+'.csv',
                   TKN)
        to_dropbox(AUC_MAT.iloc[Splitter[i]:min(Splitter[i+1], Rows),],
                   path + 'AUC/AUC_Scores_'+str(i)+'.csv',
                   TKN)
        to_dropbox(ACC_MAT.iloc[Splitter[i]:min(Splitter[i+1], Rows),],
                   path + 'ACC/ACC_Scores_'+str(i)+'.csv',
                   TKN)
        
        if Splitter[i+1] >= Rows:
            break
    else:
        
        to_dropbox(TRSCR_DF_02.iloc[Splitter[i]+1:min(Splitter[i+1], Rows),],
                   path + 'Train_Score_LC/TraninScoreLR02_'+str(i)+'.csv',
                   TKN)
        to_dropbox(TRSCR_DF_05.iloc[Splitter[i]+1:min(Splitter[i+1], Rows),],
                   path + 'Train_Score_LC/TraninScoreLR05_'+str(i)+'.csv',
                   TKN)
        to_dropbox(TRSCR_DF_09.iloc[Splitter[i]+1:min(Splitter[i+1], Rows),],
                   path + 'Train_Score_LC/TraninScoreLR09_'+str(i)+'.csv',
                   TKN)
        
        to_dropbox(TESCR_DF_02.iloc[Splitter[i]+1:min(Splitter[i+1], Rows),],
                   path + 'Test_Score_LC/TraninScoreLR02_'+str(i)+'.csv',
                   TKN)
        to_dropbox(TESCR_DF_05.iloc[Splitter[i]+1:min(Splitter[i+1], Rows),],
                   path + 'Test_Score_LC/TraninScoreLR05_'+str(i)+'.csv',
                   TKN)
        to_dropbox(TESCR_DF_09.iloc[Splitter[i]+1:min(Splitter[i+1], Rows),],
                   path + 'Test_Score_LC/TraninScoreLR09_'+str(i)+'.csv',
                   TKN)
        
        to_dropbox(configurations.iloc[Splitter[i]+1:min(Splitter[i+1], Rows),],
                   path + 'Configurations/Arquitectura_'+str(i)+'.csv',
                   TKN)
        to_dropbox(AUC_MAT.iloc[Splitter[i]+1:min(Splitter[i+1], Rows),],
                   path + 'AUC/AUC_Scores_'+str(i)+'.csv',
                   TKN)
        to_dropbox(ACC_MAT.iloc[Splitter[i]+1:min(Splitter[i+1], Rows),],
                   path + 'ACC/ACC_Scores_'+str(i)+'.csv',
                   TKN)
        
        if Splitter[i+1] >= Rows:
            break


to_dropbox(TSZ_LC_V02_DF,
           path + 'Tamano_learning_rate.csv',
           TKN)