import matplotlib.pyplot as plt
import numpy as np

# 1. Plot a histogram
data = np.random.randn(1000)

plt.figure(figsize=(8, 6))
plt.hist(data, bins=30, color='skyblue', edgecolor='black')
plt.title('Histogram of Random Data')
plt.xlabel('Value')
plt.ylabel('Frequency')
plt.savefig('1_histogram.png')
plt.close()

# 2. Create a scatter plot
x_scatter = np.random.rand(50)
y_scatter = np.random.rand(50)

plt.figure(figsize=(8, 6))
plt.scatter(x_scatter, y_scatter, color='red', edgecolor='black', marker='o')
plt.title('Random Scatter Plot')
plt.xlabel('X Value')
plt.ylabel('Y Value')
plt.savefig('2_scatter.png')
plt.close()

# 3. Create a line chart with legend
x_line = np.linspace(0, 10, 100)
y_line = np.sin(x_line)

plt.figure(figsize=(8, 6))
plt.plot(x_line, y_line, label='sin(x)', color='green', linewidth=2, linestyle='--')
plt.title('Line Chart of sin(x)')
plt.xlabel('X')
plt.ylabel('sin(x)')
plt.legend(loc='upper right')
plt.grid(True)
plt.savefig('3_line.png')
plt.close()

# 4. Create a bar chart
categories = ['A', 'B', 'C', 'D', 'E']
values = [3, 7, 2, 5, 4]

plt.figure(figsize=(8, 6))
plt.bar(categories, values, color='orange', edgecolor='black')
plt.title('Bar Chart of Categories')
plt.xlabel('Category')
plt.ylabel('Value')
plt.savefig('4_bar.png')
plt.close()

print("Success! All 4 charts have been saved as PNG files.")

