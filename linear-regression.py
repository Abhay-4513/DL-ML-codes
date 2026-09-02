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