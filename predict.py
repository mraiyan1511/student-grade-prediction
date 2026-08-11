import pandas as pd
import joblib # load

model = joblib.load('student_model.pkl')   # Load the saved model
data_df = pd.read_csv('test_data.csv')   # Load Test Data
x = data_df
y_pred = model.predict(x)  # Predict
print("Predictions:", y_pred)

