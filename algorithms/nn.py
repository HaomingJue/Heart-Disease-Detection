from sklearn.neighbors import KNeighborsClassifier


class NN:
    def __init__(self, p = 2, label = 'NN, euclidean distance'):
        '''
        Default settings:
        p=2 makes it euclidean distance
        '''
        self.name = "NN"
        self.p = p
        self.label = label
    
    def get_label(self):
        return self.label
    
    def set_label(self, label):
        self.label = label
    
    def generate_model(self, X_train, y_train):
        return KNeighborsClassifier(n_neighbors=1, p=self.p).fit(X_train,y_train)
    
        