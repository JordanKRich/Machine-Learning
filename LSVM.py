import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import time

#Load the data
train_data = pd.read_csv('spam_train.csv')
test_data = pd.read_csv('spam_test.csv')

#Drop the 'Unnamed: 0' column if it exists
if 'Unnamed: 0' in train_data.columns:
    train_data = train_data.drop('Unnamed: 0', axis=1)
if 'Unnamed: 0' in test_data.columns:
    test_data = test_data.drop('Unnamed: 0', axis=1)

#Separate features and target variable
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

#Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#Start time
start_time = time.time()

#Define the range of C values to test
C_values = [0.01, 0.1, 1, 10, 100, 200, 250, 271, 300]
cv_scores = []  # Cross-validation scores

for C in C_values:
    svm = LinearSVC(C=C, max_iter=10000)
    scores = cross_val_score(svm, X_train_scaled, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

# Find the optimal C
optimal_C = C_values[np.argmax(cv_scores)]
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Optimal C: {optimal_C}')
print(f'Optimal C Cross-Validation Accuracy: {max(cv_scores):.4f}')

# Calculate total time taken
total_time = time.time() - start_time
print(f'Total time taken: {total_time:.2f} seconds')

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(C_values, cv_scores, marker='o')
plt.xscale('log')  # Logarithmic scale for C
plt.title('Linear SVM: Cross-Validation Accuracy vs. C')
plt.xlabel('Regularization Parameter C')
plt.ylabel('Cross-Validation Accuracy')
plt.show()

# Train the final model with optimal C
svm_final = LinearSVC(C=optimal_C, max_iter=10000)
svm_final.fit(X_train_scaled, y_train)

# Predict on the training set
y_train_pred = svm_final.predict(X_train_scaled)
training_accuracy = accuracy_score(y_train, y_train_pred)
print(f'Linear SVM Training Accuracy: {training_accuracy:.4f}')

# Predict on the test set
y_pred = svm_final.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'Linear SVM Test Accuracy: {test_accuracy:.4f}')

# Analyze overfitting or underfitting
if training_accuracy > test_accuracy + 0.05:
    print("The Linear SVM model may be overfitting.")
elif test_accuracy > training_accuracy + 0.05:
    print("The Linear SVM model may be underfitting.")
else:
    print("The Linear SVM model generalizes well.")

# Create a DataFrame with true and predicted labels
results_df = pd.DataFrame({
    'Index': y_test.index,
    'True Label': y_test.values,
    'Predicted Label': y_pred
})

# Identify misclassified samples
misclassified_df = results_df[results_df['True Label'] != results_df['Predicted Label']]

# Print total number of misclassified samples
print(f"Total misclassified samples: {len(misclassified_df)}")

# Print details of the first 5 misclassified samples
print('Details of the first 5 misclassified samples:')
print(misclassified_df.head(5))
