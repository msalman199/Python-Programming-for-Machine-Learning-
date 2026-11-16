import numpy as np

# 1. Create a 1D NumPy array
arr = np.array([1, 2, 3, 4, 5])
print("1D NumPy Array:")
print(arr)

# 2. Create a 2D NumPy array (matrix)
matrix = np.array([[1, 2, 3], [4, 5, 6]])
print("\n2D NumPy Matrix:")
print(matrix)

# 3. Create a 3x3 matrix of zeros
zeros_matrix = np.zeros((3, 3))
print("\n3x3 Matrix of Zeros:")
print(zeros_matrix)

# 4. Create a 3x3 matrix of ones
ones_matrix = np.ones((3, 3))
print("\n3x3 Matrix of Ones:")
print(ones_matrix)

# 5. Array addition and subtraction
arr1 = np.array([1, 2, 3])
arr2 = np.array([4, 5, 6])

print("\nArray Addition:")
print(arr1 + arr2)

print("\nArray Subtraction:")
print(arr1 - arr2)

# 6. Element-wise multiplication
print("\nElement-wise Multiplication:")
print(arr1 * arr2)

# 7. Matrix multiplication (dot product)
matrix1 = np.array([[1, 2], [3, 4]])
matrix2 = np.array([[5, 6], [7, 8]])

print("\nMatrix Dot Product (Multiplication):")
print(np.dot(matrix1, matrix2))

# 8. Element-wise square
print("\nElement-wise Square:")
print(np.square(arr1))

# 9. Element-wise square root
print("\nElement-wise Square Root:")
print(np.sqrt(arr1))

# 10. Element-wise exponentiation
print("\nElement-wise Exponentiation (e^x):")
print(np.exp(arr1))
