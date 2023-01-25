from sklearn.tree import DecisionTreeClassifier

class GBT:
    def __init__(self, label = 'Gradient Boosted Decision Trees'):
        self.label = label
    
    def get_label(self):
        return self.label
    
    def set_label(self, label):
        self.label = label
    
    def generate_model(self, X_train, y_train):
        return DecisionTreeClassifier().fit(X_train,y_train)
    
        