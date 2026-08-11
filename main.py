
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import joblib

# Load Dataset
df = pd.read_csv("student-mat.csv")

print("Original Data Loaded:", df.shape)

# Data Cleaning
df = df.drop_duplicates()
df = df.fillna(df.median(numeric_only=True))
df = df.fillna(df.mode().iloc[0])

print("After Cleaning:", df.shape)

# Encode Categorical Columns
encoders = {}

for c in df.select_dtypes(include="object").columns:
    encoder = LabelEncoder()
    df[c] = encoder.fit_transform(df[c])
    encoders[c] = encoder

print("After Encoding:")
print(df.head())

# Target Column
target = "G3"

# Feature Selection
corr = df.corr()[target].abs()

features = corr[corr > 0.1].index.drop(target)

print("Selected Features:")
print(list(features))

# Split Data
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train Model
model = DecisionTreeClassifier(random_state=42)

model.fit(X_train, y_train)

# Prediction
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy: {:.2f}%".format(accuracy * 100))

# Save everything required by Streamlit
model_data = {
    "model": model,
    "features": list(features),
    "encoders": encoders
}

joblib.dump(model_data, "student_model.pkl")

print("student_model.pkl file created successfully!")
