import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load the dataset
df = pd.read_csv("Iris.csv")

# Convert all column names to lowercase to avoid errors
df.columns = df.columns.str.strip().str.lower()

# Display the first few rows
print(df.head())

# Drop the ID column (if present)
df.drop(columns=["id"], inplace=True, errors="ignore")  # Use lowercase "id"

# Convert species names into numbers
le = LabelEncoder()
print("Column names in the dataset:", df.columns)
df["species"] = le.fit_transform(df["species"])  # Use lowercase "species"

# Split data into features (X) and labels (y)
X = df.drop("species", axis=1)
y = df["species"]

# Split into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Check if preprocessing was successful
print("Data preprocessing completed successfully!")
print("Training samples:", X_train.shape[0])
print("Testing samples:", X_test.shape[0])

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Test the model
y_pred = model.predict(X_test)

# Print accuracy and classification report
print("\nModel Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))


import pickle

# Save the trained model
with open("iris_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved successfully as iris_model.pkl!")

