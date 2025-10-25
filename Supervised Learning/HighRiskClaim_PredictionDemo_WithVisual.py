
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Load training data
train_df = pd.read_csv("HighRisk_TrainingData.csv")

# Encode categorical variables
le_claim = LabelEncoder()
le_region = LabelEncoder()
train_df['ClaimType_enc'] = le_claim.fit_transform(train_df['ClaimType'])
train_df['Region_enc'] = le_region.fit_transform(train_df['Region'])

# Select features and target
X = train_df[['ClaimType_enc', 'Amount', 'Region_enc', 'DaysToSettle']]
y = train_df['HighRisk']

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Classification Report on Test Data:")
print(classification_report(y_test, y_pred))

# Visualize feature importance
importances = model.feature_importances_
feature_names = ['ClaimType', 'Amount', 'Region', 'DaysToSettle']
plt.figure(figsize=(8, 5))
plt.bar(feature_names, importances, color='skyblue')
plt.title('Feature Importance in Predicting High-Risk Claims')
plt.ylabel('Importance')
plt.tight_layout()
plt.savefig("FeatureImportance_HighRisk.png")
plt.show()

# Predict on new data
new_df = pd.read_csv("NewClaims_ToPredict_50Rows.csv")
new_df['ClaimType_enc'] = le_claim.transform(new_df['ClaimType'])
new_df['Region_enc'] = le_region.transform(new_df['Region'])

X_new = new_df[['ClaimType_enc', 'Amount', 'Region_enc', 'DaysToSettle']]
new_df['Predicted_HighRisk'] = model.predict(X_new)

# Show results
print("\nPredictions on New Claims:")
print(new_df[['ClaimType', 'Amount', 'Region', 'DaysToSettle', 'Predicted_HighRisk']])

# Save predictions
new_df.to_csv("Predicted_HighRisk_Claims.csv", index=False)
