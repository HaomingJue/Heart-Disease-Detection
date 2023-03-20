from sklearn.ensemble import StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn.ensemble import GradientBoostingClassifier


class Stacked:
    def __init__(self, label = 'Stacked Classifier'):
        self.name = "Stacking"
        self.label = label
    
    def get_label(self):
        return self.label
    
    def set_label(self, label):
        self.label = label
    
    def generate_model(self, X_train, y_train):
        ensemble = [('gbtclass', GradientBoostingClassifier(max_depth=4)), ('knnclass', KNeighborsClassifier(n_neighbors=5, p=2)), ('nbclass', GaussianNB()), ('nnclass',  KNeighborsClassifier(n_neighbors=1, p=2)), ('svmlinclass', svm.SVC(kernel="linear", C=0.5, probability=True)), ('svmrbfclass', svm.SVC(kernel="rbf", C=0.5, gamma=1, probability=True))]
        return StackingClassifier(estimators = ensemble).fit(X_train, y_train)
    


