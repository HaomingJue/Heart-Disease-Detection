import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn import metrics
from collections import Counter  
import seaborn as sns
from matplotlib import pyplot as plt

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def prepare_dataset(train_id):
    #Import data
    #df1 = pd.read_csv("processedhungarian.csv")
    #df1.columns = ["Age", "Sex", "Chest Muscles Pain Variety", "Relaxing Blood Force", "Serum Cholesterol", "Fasting Blood Glucose Level", "Resting ECG Benefits", "Maximum Pulse Rate", "Exercise Induced Angina", "ST Depressive Disorder",  "Slope in Peak Exercising ST Message", "Fluoroscopy", "Thalium", "ClassID"] 
    #df1 = df1.drop([ "Relaxing Blood Force", "Fasting Blood Glucose Level", "Resting ECG Benefits"], axis=1) #dropping these columns because the paper said they were the least important

    #df2 = pd.read_csv("processedcleveland.csv")
    #df2.columns = ["Age", "Sex", "Chest Muscles Pain Variety", "Relaxing Blood Force", "Serum Cholesterol", "Fasting Blood Glucose Level", "Resting ECG Benefits", "Maximum Pulse Rate", "Exercise Induced Angina", "ST Depressive Disorder",  "Slope in Peak Exercising ST Message", "Fluoroscopy", "Thalium", "ClassID"] 
    #df2 = df2.drop([ "Relaxing Blood Force", "Fasting Blood Glucose Level", "Resting ECG Benefits"], axis=1) #dropping these columns because the paper said they were the least important

    #df = df1.append(df2, ignore_index=False, sort=None)
    
    df = pd.read_csv("heart.csv")
    #df.columns = ["Age", "Sex", "Chest Muscles Pain Variety", "Relaxing Blood Force", "Serum Cholesterol", "Fasting Blood Glucose Level", "Resting ECG Benefits", "Maximum Pulse Rate", "Exercise Induced Angina", "ST Depressive Disorder",  "Slope in Peak Exercising ST Message", "Fluoroscopy", "Thalium", "ClassID"] 
    #df = df1.drop([ "Relaxing Blood Force", "Fasting Blood Glucose Level", "Resting ECG Benefits"], axis=1) #dropping these columns because the paper said they were the least important


    #Replace missing values with the mean
    imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = imp.fit(df.iloc[:,0:12])
    df.iloc[:,0:12] = pd.DataFrame(imp.transform(df.iloc[:,0:12]))

            
    allData = df

    #Split Answers from collected data
    X = allData.iloc[:, 0:13]
    y = allData.iloc[:, 13]

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
        ax = fig.add_subplot(4, 4, i+1)
        sns.histplot(traindat, x=traindat.columns[i], hue="target", ax=ax, bins=30)
        ax.set(title=df.columns[i], xlabel='Value')
        ax.legend(loc= 'upper right', labels=['Sick', 'Healthy'])
        fig.tight_layout()
        
    fig.savefig(f"results/{train_id}/figures/Class Distribution")

    return X_train, X_test, y_train, y_test