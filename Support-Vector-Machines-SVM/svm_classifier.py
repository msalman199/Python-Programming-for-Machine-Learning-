from sklearn import datasets
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.svm import SVC

# 1. Load the Iris dataset
iris = datasets.load_iris()
X = iris.data  # Features
y = iris.target  # Target labels

# Split the dataset into training and testing sets (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Train a baseline SVM classifier with a linear kernel
model = SVC(kernel='linear', random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f'Baseline Model Accuracy: {accuracy * 100:.2f}%')
print('-' * 50)

# 3. Tune SVM hyperparameters using GridSearchCV
param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto'],
}

svm_base = SVC(random_state=42)
grid_search = GridSearchCV(
    svm_base, param_grid, cv=5, scoring='accuracy', verbose=1
)
grid_search.fit(X_train, y_train)

print('Best Parameters:', grid_search.best_params_)
print('Best Cross-validation Accuracy:', grid_search.best_score_)

# Evaluate the best tuned model on the test set
best_model = grid_search.best_estimator_
y_pred_tuned = best_model.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
print(f'Model Accuracy after tuning: {tuned_accuracy * 100:.2f}%')
