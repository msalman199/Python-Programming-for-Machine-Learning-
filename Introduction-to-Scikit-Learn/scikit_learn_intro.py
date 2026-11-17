import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

# 1. Load the California housing dataset
data = fetch_california_housing()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print('First 5 rows of the dataset:')
print(df.head())
print('-' * 50)

# 2. Split data and train Linear Regression model
X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f'Model Coefficients: {model.coef_}')
print(f'Model Intercept: {model.intercept_}')
print('-' * 50)

# 3. Evaluate model performance
mse = mean_squared_error(y_test, y_pred)
r2_score = model.score(X_test, y_test)

print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared score: {r2_score}')
