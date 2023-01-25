import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn import metrics
from collections import Counter  

class DecisionTree:
    def __init__(self, max_depth = 4, label = 'Decision Tree'):
        self.max_depth = max_depth
        self.label = label + f' ,max depth = self.max_depth'
    
    def get_label(self):
        return self.label
    
    def set_label(self, label):
        self.label = label
    
    def generate_model(self, X_train, y_train):
        return DecisionTreeClassifier(max_depth=self.max_depth).fit(X_train,y_train)
    
        