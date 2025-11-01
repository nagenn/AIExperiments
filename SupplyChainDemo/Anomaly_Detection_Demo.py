
# Anomaly Detection Demo: Spotting Delivery Disruptions
# Works in both Google Colab and VS Code

# Install dependencies (if needed)
#try:
#    import sklearn
#except ImportError:
#    !pip install scikit-learn

#try:
#    import matplotlib
#except ImportError:
#    !pip install matplotlib

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Load the data
df = pd.read_csv("Shipment_Delivery_Data.csv")

# Select features for anomaly detection
features = df[['Distance_km', 'DeliveryTime_hr']]

# Fit Isolation Forest
model = IsolationForest(contamination=0.02, random_state=42)
df['Anomaly'] = model.fit_predict(features)

# Convert anomaly flag: -1 = anomaly, 1 = normal
df['Anomaly'] = df['Anomaly'].map({1: 'Normal', -1: 'Anomaly'})

# Print summary
print("Total Records:", len(df))
print("Detected Anomalies:", (df['Anomaly'] == 'Anomaly').sum())

# Display anomaly records
anomalies = df[df['Anomaly'] == 'Anomaly']
print("\nTop 10 Anomalies:")
print(anomalies[['ShipmentID', 'Origin', 'Destination', 'Distance_km', 'DeliveryTime_hr']].head(10))

# Plot
plt.figure(figsize=(10, 6))
normal = df[df['Anomaly'] == 'Normal']
anomaly = df[df['Anomaly'] == 'Anomaly']

plt.scatter(normal['Distance_km'], normal['DeliveryTime_hr'], c='blue', label='Normal', alpha=0.6)
plt.scatter(anomaly['Distance_km'], anomaly['DeliveryTime_hr'], c='red', label='Anomaly', marker='x', s=100)
plt.xlabel("Distance (km)")
plt.ylabel("Delivery Time (hr)")
plt.title("Anomaly Detection in Delivery Times")
plt.legend()
plt.grid(True)
plt.show()
