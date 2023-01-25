# import pickle
# import joblib
# import pandas as pd
# import numpy as np
# from sklearn.model_selection import train_test_split
# from sklearn import svm
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.model_selection import cross_val_score
# from sklearn.metrics import classification_report
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import precision_score
# from sklearn.metrics import confusion_matrix
# from sklearn.naive_bayes import GaussianNB
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.impute import SimpleImputer
# from sklearn import metrics
# from collections import Counter  
# import seaborn as sns
# from matplotlib import pyplot as plt

# test_data = [[8.0,1.0,4.0,355.0,99.0,1.0,2.0,2.0,1,2]]
# load_model = joblib.load(open("model.sav", 'rb'))
# y_pred = load_model.predict(test_data)

# print(y_pred)

import os
from datetime import datetime
fuck = "fuckfuck"

train_id = datetime.now().strftime("%m_%d_%Y__%H_%M_%S")
syss = f"cd results && mkdir {train_id}"
os.system(syss)