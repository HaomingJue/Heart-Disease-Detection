from sklearn.neighbors import KNeighborsClassifier

class KNN:
    def __init__(self, k = 5, p = 2, label = 'kNN, k=5, euclidean distance'):
        '''
        Default settings:
        p=2 makes it euclidean distance
        '''
        self.k = k
        self.p = p
        self.label = label
    
    def get_label(self):
        return self.label
    
    def set_label(self, label):
        self.label = label
    
    def generate_model(self, X_train, y_train):
        return KNeighborsClassifier(n_neighbors=self.k, p=self.p).fit(X_train,y_train)
    
        