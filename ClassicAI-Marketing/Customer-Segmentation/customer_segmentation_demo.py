import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np

# Load dataset from CSV
df = pd.read_csv("customer_segmentation_dataset.csv")

# Select features for clustering
X = df[['Annual_Spend', 'Visits_per_Month', 'Engagement_Score']]

# Find optimal number of clusters using Elbow Method
inertia = []
k_range = range(2, 11)  # Test between 2 and 10 clusters
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X)
    inertia.append(kmeans.inertia_)

# Find elbow point automatically
deltas = np.diff(inertia)
second_deltas = np.diff(deltas)
optimal_k = k_range[np.argmax(second_deltas) + 2]

print(f"Optimal number of clusters: {optimal_k}")

# Run KMeans clustering with optimal number of segments
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['Segment'] = kmeans.fit_predict(X)

# Dynamically assign named colors based on number of segments
num_segments = df['Segment'].nunique()
named_colors = list(mcolors.TABLEAU_COLORS.keys())
color_map = {i: named_colors[i].replace('tab:', '') for i in range(num_segments)}

df['Segment_Color'] = df['Segment'].map(color_map)

# Summary with color name as index
summary = df.groupby('Segment').agg(
    Total_Customers=('CustomerID', 'count'),
    Avg_Annual_Spend=('Annual_Spend', 'mean'),
    Avg_Engagement_Score=('Engagement_Score', 'mean')
)
summary.index = summary.index.map(color_map)
summary.index.name = 'Segment'
print(summary)

# Visualize segmentation
plt.scatter(df['Annual_Spend'], df['Engagement_Score'], c=df['Segment_Color'])
plt.xlabel('Annual Spend')
plt.ylabel('Engagement Score')
plt.title('Customer Segmentation with Classic AI (K-means)')
plt.show()
