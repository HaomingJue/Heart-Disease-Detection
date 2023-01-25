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
    class0.to_csv(f'results/{train_id}/dataset/class0.csv', index=False)
    class1 = class1.transpose()
    class1.to_csv(f'results/{train_id}/dataset/class1.csv', index=False)

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
        
    fig.savefig(f"results/{train_id}/figures/Class Distribution")

    return X_train, X_test, y_train, y_test