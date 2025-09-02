
import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

# Load dataset from CSV
df = pd.read_csv("customer_segmentation_dataset.csv")

# Select features for clustering
X = df[['Annual_Spend', 'Visits_per_Month', 'Engagement_Score']]

# Run KMeans clustering (3 segments)
kmeans = KMeans(n_clusters=3, random_state=42)
df['Segment'] = kmeans.fit_predict(X)

# Show clustered data
print(df.head())

# Visualize segmentation
plt.scatter(df['Annual_Spend'], df['Engagement_Score'], c=df['Segment'], cmap='viridis')
plt.xlabel('Annual Spend')
plt.ylabel('Engagement Score')
plt.title('Customer Segmentation with Classic AI (K-means)')
plt.show()
