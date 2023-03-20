from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier


class Bagged:
    def __init__(self, label = 'Bagging Classifier'):
        self.name = "Bagging"
        self.label = label
    
    def get_label(self):
        return self.label
    
    def set_label(self, label):
        self.label = label
    
    def generate_model(self, X_train, y_train):
        return BaggingClassifier(GaussianNB(), n_estimators=10, random_state=0).fit(X_train, y_train)
    