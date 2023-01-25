import joblib

test_data = [[8.0,1.0,4.0,355.0,99.0,1.0,2.0,2.0,1,2]]
load_model = joblib.load(open("results/01_25_2023__01_14_12/models/NN.joblib", 'rb'))
y_pred = load_model.predict(test_data)

print(y_pred)
