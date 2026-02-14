# -------------------------------
# 1. Import libraries
# -------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# -------------------------------
# 2. Load dataset
# -------------------------------
df = pd.read_csv("dataset.csv")
print("Dataset loaded successfully!")
print("Columns:", df.columns.tolist())

# -------------------------------
# 3. Select target column
# -------------------------------
target = input("Enter target column name: ")

if target not in df.columns:
    print("Invalid target column!")
    exit()

# -------------------------------
# 4. Separate features and target
# -------------------------------
X = df.drop(columns=[target])
y = df[target]

# -------------------------------
# 5. Split dataset
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 6. Train Naive Bayes model
# -------------------------------
model = GaussianNB()
model.fit(X_train, y_train)

# -------------------------------
# 7. Make predictions
# -------------------------------
y_pred = model.predict(X_test)

# -------------------------------
# 8. Evaluate model
# -------------------------------
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
