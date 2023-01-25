from sklearn import svm

class SVMLinear:
    def __init__(self, C = 0.5, label = 'SVM linear'):
        self.kernal = 'linear'
        self.C = C
        self.label = label + f' ,C = {self.C}'
    
    def get_label(self):
        return self.label
    
    def set_label(self, label):
        self.label = label
    
    def generate_model(self, X_train, y_train):
        return svm.SVC(kernel=self.kernal, C=self.C, probability=True).fit(X_train, y_train)
    
        