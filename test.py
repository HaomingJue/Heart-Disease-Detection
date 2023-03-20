import joblib
import pandas as pd


trained_time = "03_19_2023__22_31_48"
model = "NaiveBayes"
test_data = pd.DataFrame({
    'age': [1],
    'sex': [2],
    'cp': [3],
    'trestbps': [4],
    'chol': [5],
    'fbs': [6],
    'restecg': [7],
    'thalach': [8],
    'exang': [9],
    'oldpeak': [10],
    'slope': [11],
    'ca': [12],
    'thal': [13]
}) if model == "XGB" else [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11 ,12 ,13]]
load_model = joblib.load(open(f"results/{trained_time}/models/{model}.joblib", 'rb'))
y_pred = load_model.predict(test_data)

print(y_pred)