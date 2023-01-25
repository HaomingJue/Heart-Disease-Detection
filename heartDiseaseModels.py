# -*- coding: utf-8 -*-
"""
Created on Thu Jan 19 16:35:11 2023

@author: Alexis

This is the intial version
test
"""
import pickle
import joblib
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
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

#Import data
df = pd.read_csv("processedhungarian.csv")
df.columns = ["Age", "Sex", "Chest Muscles Pain Variety", "Relaxing Blood Force", "Serum Cholesterol", "Fasting Blood Glucose Level", "Resting ECG Benefits", "Maximum Pulse Rate", "Exercise Induced Angina", "ST Depressive Disorder",  "Slope in Peak Exercising ST Message", "Fluoroscopy", "Thalium", "ClassID"] 
df = df.drop([ "Relaxing Blood Force", "Fasting Blood Glucose Level", "Resting ECG Benefits"], axis=1) #dropping these columns because the paper said they were the least important

#separates the class into different data frames
class0 = pd.DataFrame()
class1 = pd.DataFrame()

for i in range(0, len(df)):
    if df.iloc[i,10] == 0:
        if class0.empty:
            class0 = df.iloc[i]
            class0 = class0.to_frame()
        else: 
            class0 = class0.merge(df.iloc[i].to_frame(), right_index=True, left_index=True)
    if df.iloc[i,10] == 1:
        if class1.empty:
            class1 = df.iloc[i]
            class1 = class1.to_frame()
        else: 
            class1 = class1.merge(df.iloc[i].to_frame(), right_index=True, left_index=True)

class0 = class0.transpose()
class0.to_csv('class0.csv', index=False)
class1 = class1.transpose()
class1.to_csv('class1.csv', index=False)

#Replace missing values with the mean
imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(class0.iloc[:,0:10])
class0.iloc[:,0:10] = pd.DataFrame(imp.transform(class0.iloc[:,0:10]), columns = ["Age", "Sex", "Chest Muscles Pain Variety", "Serum Cholesterol", "Maximum Pulse Rate", "Exercise Induced Angina", "ST Depressive Disorder",  "Slope in Peak Exercising ST Message", "Fluoroscopy", "Thalium"])

imp = SimpleImputer(missing_values=np.nan, strategy='mean')
imp = imp.fit(class1.iloc[:,0:10])
class1.iloc[:,0:10] = pd.DataFrame(imp.transform(class1.iloc[:,0:10]), columns = ["Age", "Sex", "Chest Muscles Pain Variety", "Serum Cholesterol", "Maximum Pulse Rate", "Exercise Induced Angina", "ST Depressive Disorder",  "Slope in Peak Exercising ST Message", "Fluoroscopy", "Thalium"])
         
allData = class0.append(class1, ignore_index=False, sort=None) 

#Split Answers from collected data
X = allData.iloc[:, 0:10]
y = allData.iloc[:, 10]

# Split dataset 
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2) 
# Report the number of samples from each class in your train and test sets
print("Train:", Counter(y_train))
print("Test:", Counter(y_test))
traindat = pd.concat([X_train, y_train], axis=1)


# Graphing Distributions
sns.set(style = "darkgrid", font_scale=0.5)
fig = plt.figure()
fig.suptitle("Feature Distribution")


for i in range(len(traindat.iloc[1])):
    ax = fig.add_subplot(4, 3, i+1)
    sns.histplot(traindat, x=traindat.columns[i], hue="ClassID", ax=ax, bins=30)
    ax.set(title=df.columns[i], xlabel='Value')
    ax.legend(loc= 'upper right', labels=['Sick', 'Healthy'])
    fig.tight_layout()
    
fig.savefig("Class Distribution")

#SVM, 5-fold cross validation, linear kernel, C=1
svmlinC1 = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
scores = cross_val_score(svmlinC1, X_train, y_train, cv=5)

for i in range(0,5):
    print("The accuracy for fold", i,"is" , scores[i])

classifier_predacc = scores.mean()
print("The average accuracy across all five folds is:", classifier_predacc)
classifier_predstd = scores.std()
print("The standard deviation across the five accuracy measurements is:", classifier_predstd)


svmlinC1_2 = svm.SVC(kernel='linear', C=1).fit(X_train, y_train)
y_pred = svmlinC1_2.predict(X_test)
classifier_acc = accuracy_score(y_test, y_pred)
print("The accuracy of this model is:",classifier_acc)
if classifier_acc > classifier_predacc+classifier_predstd or classifier_acc < classifier_predacc-classifier_predstd:
    print("The accuracy does not fall within 1 standard deviation of the predicted accuracy.")
else:
    print("The accuracy falls within 1 standard deviation of the predicted accuracy.")

#ROC Curves for classifiers
plt.figure(0).clf()
plt.title("ROC Curves for Six Difference Classifiers.")


#kNN, K=5, euclidean distance:
knn5 = KNeighborsClassifier(n_neighbors=5, p=2).fit(X_train,y_train) #p=2 makes it euclidean distance
y_pred_knn5 = knn5.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_knn5)
auc = round(metrics.roc_auc_score(y_test, y_pred_knn5), 4)
scores = cross_val_score(knn5, X_train, y_train, cv=5)
plt.plot(fpr,tpr,label="kNN, k=5, euclidean distance, Average Accuracy: ="+str(scores.mean())+"+/-"+str(scores.std()),linewidth=0.5)

#NN, euclidean distance
nn = KNeighborsClassifier(n_neighbors=1, p=2).fit(X_train,y_train) #p=2 makes it euclidean distance
y_pred_nn = nn.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_nn)
auc = round(metrics.roc_auc_score(y_test, y_pred_nn), 4)
scores = cross_val_score(nn, X_train, y_train, cv=5)
plt.plot(fpr,tpr,label="NN, euclidean distance, Average Accuracy: ="+str(scores.mean())+"+/-"+str(scores.std()),linewidth=0.5)

#Naive Bayes
NB = GaussianNB().fit(X_train,y_train)
y_pred_NB = NB.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_NB)
auc = round(metrics.roc_auc_score(y_test, y_pred_NB), 4)
scores = cross_val_score(NB, X_train, y_train, cv=5)
plt.plot(fpr,tpr,label="Naive Bayes, Average Accuracy: ="+str(scores.mean())+"+/-"+str(scores.std()),linewidth=0.5)

#SVM linear kernal, C=0.5
svmlinC05 = svm.SVC(kernel='linear', C=0.5, probability=True).fit(X_train, y_train)
y_pred_svmlinC05 = svmlinC05.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_svmlinC05)
auc = round(metrics.roc_auc_score(y_test, y_pred_svmlinC05), 4)
scores = cross_val_score(svmlinC05, X_train, y_train, cv=5)
plt.plot(fpr,tpr,label="SVM, linear kernel, C=0.5, Average Accuracy: ="+str(scores.mean())+"+/-"+str(scores.std()),linewidth=0.5)

#SVM with RBF/gaussian kernel, C=0.5, gamma = 1
svmrbfC05g1 = svm.SVC(kernel='rbf', C=0.5, gamma=1, probability=True).fit(X_train, y_train)
y_pred_svmrbfC05g1 = svmrbfC05g1.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_svmrbfC05g1)
auc = round(metrics.roc_auc_score(y_test, y_pred_svmrbfC05g1), 4)
scores = cross_val_score(svmrbfC05g1, X_train, y_train, cv=5)
plt.plot(fpr,tpr,label="SVM, RBF, C=0.5, Average Accuracy: ="+str(scores.mean())+"+/-"+str(scores.std()),linewidth=0.5)

#decision tree with max depth of 4
dt = DecisionTreeClassifier(max_depth=4).fit(X_train,y_train)
y_pred_dt = dt.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_dt)
auc = round(metrics.roc_auc_score(y_test, y_pred_dt), 4)
scores = cross_val_score(dt, X_train, y_train, cv=5)
plt.plot(fpr,tpr,label="Decision Tree, max depth=4, Average Accuracy: ="+str(scores.mean())+"+/-"+str(scores.std()),linewidth=0.5)

    #GBT with max depth of 4
gbt = DecisionTreeClassifier().fit(X_train,y_train)
y_pred_gbt = gbt.predict_proba(X_test)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_pred_gbt)
auc = round(metrics.roc_auc_score(y_test, y_pred_gbt), 4)
scores = cross_val_score(gbt, X_train, y_train, cv=5)
plt.plot(fpr,tpr,label="Gradient Boosted Decision Trees, Average Accuracy: ="+str(scores.mean())+"+/-"+str(scores.std()),linewidth=0.5)

filename = 'model.joblib'
joblib.dump(gbt, open(filename, 'wb'))

with open('model.pkl', 'wb') as f:
    pickle.dump(gbt, f)

    
plt.legend()
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
    
plt.savefig("ROCCurves") 