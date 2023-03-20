from xgboost import XGBClassifier

class XGB:
    def __init__(self, label = 'XGBoost Decision Trees'):
        self.name = "XGB"
        self.label = label
    
    def get_label(self):
        return self.label
    
    def set_label(self, label):
        self.label = label
    
    def generate_model(self, X_train, y_train):
        return XGBClassifier(objective="binary:logistic", random_state=42).fit(X_train,y_train)
    
        