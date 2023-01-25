from sklearn import svm

class SVMRBF:
    def __init__(self, C = 0.5, gamma = 1, label = 'SVM, RBF'):
        self.kernal = 'rbf'
        self.C = C
        self.gamma = gamma
        self.label = label + f' ,C = {self.C}'
    
    def get_label(self):
        return self.label
    
    def set_label(self, label):
        self.label = label
    
    def generate_model(self, X_train, y_train):
        return svm.SVC(kernel=self.kernal, C=self.C, gamma=self.gamma, probability=True).fit(X_train, y_train)
    
        