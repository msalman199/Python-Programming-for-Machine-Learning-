import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

# 1. Load the Iris dataset
url = "iris.csv"
column_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "class"]
df = pd.read_csv(url, header=None, names=column_names)

print("First 5 rows of the dataset:")
print(df.head())

# 2. Simulate and handle missing values
df.loc[2, 'sepal_length'] = None
df.loc[5, 'petal_width'] = None

print("\nDataset with missing values:")
print(df.head(6))

# Replace NaN with the mean of the column
df['sepal_length'] = df['sepal_length'].fillna(df['sepal_length'].mean())
df['petal_width'] = df['petal_width'].fillna(df['petal_width'].mean())

print("\nDataset after handling missing values:")
print(df.head(6))

# 3. Normalize numerical data (0 to 1 range)
numerical_columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
scaler = MinMaxScaler()
df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

print("\nDataset after normalization:")
print(df.head())

# 4. Split dataset for training (80%) and testing (20%)
X = df[numerical_columns]
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)
