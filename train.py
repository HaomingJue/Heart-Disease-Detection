import os
from utils.dataaset_prep import prepare_dataset
from utils.folder_prep import prepare_folder
from datetime import datetime


#algoirhm import
from algorithms.decision_tree import DecisionTree
from algorithms.gbt import GBT
from algorithms.knn import KNN
from algorithms.naive_bayes import NaiveBayes
from algorithms.nn import NN
from algorithms.svm_linear import SVMLinear
from algorithms.svm_rbf import SVMRBF




import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn import metrics
from collections import Counter  
import seaborn as sns
from matplotlib import pyplot as plt


train_id = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
print("current training id:", train_id)



plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

prepare_folder(train_id)
X_train, X_test, y_train, y_test = prepare_dataset(train_id)


algorithm_list = []

gbt = GBT()
decision_tree = DecisionTree()
knn = KNN()

algorithm_list.append(gbt)
algorithm_list.append(decision_tree)
algorithm_list.append(knn)

for algorithm in algorithm_list:
    model = algorithm.generate_model(X_train, y_train)
    y_pred = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = metrics.roc_curve(y_test, y_pred)
    auc = round(metrics.roc_auc_score(y_test, y_pred), 4)
    scores = cross_val_score(model, X_train, y_train, cv=5)
    plt.plot(fpr,tpr,label= algorithm.get_label() +str(scores.mean())+"+/-"+str(scores.std()),linewidth=0.5)


plt.legend()
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
    
plt.savefig("ROCCurves") 

