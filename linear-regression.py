import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)
plt.style.use('seaborn-v0_8-whitegrid')

X = np.linspace(1, 10, 30)
noise = np.random.normal(0, 4, size=X.shape)
y = 32 + 6.2 * X + noise

data = pd.DataFrame({'Study Hours': X, 'Exam Score': y})
display(data.head())
print(f'Number of observations: {len(data)}')
print(data.describe().round(2))

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color='royalblue', edgecolor='black', alpha=0.8)
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Study Hours versus Exam Score')
plt.show()

def fit_analytical(x, target):
    x_mean = np.mean(x)
    y_mean = np.mean(target)
    numerator = np.sum((x - x_mean) * (target - y_mean))
    denominator = np.sum((x - x_mean) ** 2)
    slope = numerator / denominator
    intercept = y_mean - slope * x_mean
    return intercept, slope

b0_analytical, b1_analytical = fit_analytical(X, y)
y_pred_analytical = b0_analytical + b1_analytical * X

print(f'Intercept (b0): {b0_analytical:.4f}')
print(f'Slope (b1):     {b1_analytical:.4f}')
print(f'Equation: Score = {b0_analytical:.2f} + {b1_analytical:.2f} × Study Hours')

plt.figure(figsize=(8, 5))
plt.scatter(X, y, label='Actual data', color='royalblue', edgecolor='black')
plt.plot(X, y_pred_analytical, label='Best-fit line', color='crimson', linewidth=2.5)
plt.xlabel('Study Hours')
plt.ylabel('Exam Score')
plt.title('Analytical Linear Regression')
plt.legend()
plt.show()