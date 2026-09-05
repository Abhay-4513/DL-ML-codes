# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
sns.set_theme(style='whitegrid', context='notebook')
print('Libraries imported successfully.')

centers = [(28, 25), (32, 78), (68, 30), (72, 75)]
cluster_std = [6.5, 7.0, 7.5, 6.0]
X_raw, _ = make_blobs(
    n_samples=320, centers=centers, cluster_std=cluster_std,
    random_state=RANDOM_STATE
)

df = pd.DataFrame(X_raw, columns=['Annual Income (₹ lakh scaled)', 'Spending Score'])
df['Annual Income (₹ lakh scaled)'] = df['Annual Income (₹ lakh scaled)'].clip(10, 100)
df['Spending Score'] = df['Spending Score'].clip(1, 100)

print('Dataset shape:', df.shape)
display(df.head(10))


df.info()
print('\nMissing values:')
display(df.isna().sum().to_frame('Missing'))
print('Summary statistics:')
display(df.describe().round(2))

fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))
sns.histplot(df.iloc[:, 0], kde=True, ax=axes[0], color='steelblue')
axes[0].set_title('Income Distribution')
sns.histplot(df.iloc[:, 1], kde=True, ax=axes[1], color='darkorange')
axes[1].set_title('Spending Score Distribution')
sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], ax=axes[2], s=50)
axes[2].set_title('Customers Before Clustering')
plt.tight_layout()
plt.show()