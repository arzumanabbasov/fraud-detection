import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import joblib

plt.style.use('seaborn')

# Read the dataset
df = pd.read_csv('C:/Users/Admin/Downloads/PS_20174392719_1491204439457_log.csv')

# Preprocessing steps...

# Print the first few rows of the dataframe
print(df.head())

# Print information about the dataframe, including column names and data types
print(df.info())

# Generate descriptive statistics of the numerical columns in the dataframe
print(df.describe())

# Check for missing values in the dataframe
print(df.isnull().sum())

# Drop rows with missing values
df = df.dropna()

# Encode the 'type' column using label encoding
le = LabelEncoder()
df['type'] = le.fit_transform(df['type'])

# Print the first few rows of the modified dataframe
print(df.head())

# Drop the 'nameOrig', 'nameDest', and 'isFlaggedFraud' columns from the dataframe
df = df.drop(columns=['nameOrig', 'nameDest', 'isFlaggedFraud'])

# Print the first few rows of the modified dataframe
print(df.head())

# Create a distribution plot for the 'step' column
sns.displot(df.step)
plt.title('Step')
plt.savefig('plots/step.png')
plt.close()

# Create a distribution plot for the 'amount' column
sns.displot(df.amount)
plt.title('Amount')
plt.savefig('plots/amount.png')
plt.close()

# Apply logarithmic transformation to the 'amount' column
df['amount'] = np.log1p(df['amount'])  # Use np.log1p to avoid division by zero for zero values

# Print the first few rows of the modified dataframe
print(df.head())

# Balance the dataset by randomly sampling an equal number of records for each 'isFraud' category
value_counts = df['isFraud'].value_counts()
min_count = min(value_counts)
df = df.groupby('isFraud').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)

# Split the dataset into features (X) and target variable (y)
X = df.drop(columns=['isFraud'])
y = df['isFraud']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a dictionary of classifiers
classifiers = {
    'Logistic Regression': LogisticRegression(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(),
    'Decision Tree': DecisionTreeClassifier()
}

# Evaluate and tune each classifier
results = []
best_estimators = {}

for name, classifier in classifiers.items():
    # Evaluate the model using cross-validation
    scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')
    accuracy_mean = scores.mean()
    accuracy_std = scores.std()

    # Perform hyperparameter tuning
    if name == 'Logistic Regression':
        param_grid = {'C': [0.1, 1, 10]}
    elif name == 'Random Forest':
        param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 5, 10]}
    elif name == 'XGBoost':
        param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [3, 5, 7]}
    elif name == 'Decision Tree':
        param_grid = {'max_depth': [None, 5, 10]}
    else:
        raise ValueError("The Name of the Model is Not Defined")

    grid_search = GridSearchCV(classifier, param_grid=param_grid, scoring='accuracy', cv=5)
    grid_search.fit(X_train, y_train)

    # Save the best estimator
    best_estimator = grid_search.best_estimator_
    best_estimators[name] = best_estimator

    # Evaluate the best estimator on the test set
    y_pred = best_estimator.predict(X_test)
    test_accuracy = accuracy_score(y_test, y_pred)

    results.append((name, accuracy_mean, accuracy_std, test_accuracy))

    # Print the evaluation results
print("Evaluation Results:")
for result in results:
    name, accuracy_mean, accuracy_std, test_accuracy = result
    print(f"Classifier: {name}")
    print(f"Train Accuracy: {accuracy_mean:.4f} +/- {accuracy_std:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}\n")

# Perform predictions using the best estimators
for name, estimator in best_estimators.items():
    y_pred = estimator.predict(X_test)
    print(f"Predictions using {name}: {y_pred}\n")

# Saving best performed model
best_model = None
best_accuracy = 0.0

for name, result in zip(best_estimators.keys(), results):
    _, _, _, test_accuracy = result
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        best_model = best_estimators[name]

# Save the best performing model
joblib.dump(best_model, 'model/fraud_detector.joblib')
