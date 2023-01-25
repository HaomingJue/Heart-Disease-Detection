from sklearn.naive_bayes import GaussianNB


class NaiveBayes:
    def __init__(self, label = 'Naive Bayes'):
        self.name = "NaiveBayes"
        self.label = label
    
    def get_label(self):
        return self.label
    
    def set_label(self, label):
        self.label = label
    
    def generate_model(self, X_train, y_train):
        return GaussianNB().fit(X_train,y_train)
    
        