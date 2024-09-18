import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import StandardScaler

#Load data
train_data = pd.read_csv('spam_train.csv', index_col=0)
test_data = pd.read_csv('spam_test.csv', index_col=0)

#Drop Unamed:0 if it exists
if 'Unnamed: 0' in train_data.columns:
    train_data = train_data.drop('Unnamed:0', axis=1)
if 'Unnamed: 0' in test_data.columns:
    test_data = test_data.drop('Unnamed:0', axis=1)

#Separate features and target variable
X_train = train_data.drop('class', axis=1)
y_train = train_data['class']
X_test = test_data.drop('class', axis=1)
y_test = test_data['class']

#Start time
start_time = time.time()

#Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Define the range of K values to test
k_range = range(1, 20)  
cv_scores = [] #Cross-validation scores
#For each K value, perform 5-fold cross-validation
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_train, y_train, cv=5, scoring='accuracy')
    cv_scores.append(scores.mean())

#Find the optimal K
optimal_k = k_range[np.argmax(cv_scores)]
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Optimal K: {optimal_k}')
print(f'Optimal K Cross-Validation Accuracy: {max(cv_scores):.4f}')

#Calculate total time taken
total_time = time.time() - start_time
print(f'Total time taken: {total_time:.2f} seconds')

#Plotting
plt.figure(figsize=(10, 6))
plt.plot(k_range, cv_scores, marker='o')
plt.title('KNN: Cross-Validation Accuracy vs. K')
plt.xlabel('Number of Neighbors (K)')
plt.ylabel('Cross-Validation Accuracy')
plt.show()

#Train the final model with optimal K
knn_final = KNeighborsClassifier(n_neighbors=optimal_k)
knn_final.fit(X_train, y_train)

#Predict on the training set
y_train_pred = knn_final.predict(X_train)
training_accuracy = accuracy_score(y_train, y_train_pred)
print(f'KNN Training Accuracy: {training_accuracy:.4f}')

#Predict on the test set
y_pred = knn_final.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f'KNN Test Accuracy: {test_accuracy:.4f}')

#Analyze overfitting or underfitting
if training_accuracy > test_accuracy + 0.05:
    print("The KNN model may be overfitting.")
elif test_accuracy > training_accuracy + 0.05:
    print("The KNN model may be underfitting.")
else:
    print("The KNN model generalizes well.")

#Create a DataFrame with the true and predicted labels
results_df = pd.DataFrame({
    'Index': y_test.index,
    'True Label': y_test.values,
    'Predicted Label': y_pred
})

#Identify misclassified samples
misclassified_df = results_df[results_df['True Label'] != results_df['Predicted Label']]

#Print total number
print(f"Total misclassified samples: {len(misclassified_df)}")

#Print details
print('First 5 misclassified samples:')
print(misclassified_df.head(5))
