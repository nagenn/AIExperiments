This is a demo of Classic AI in action using the Isolation forest algorithm
We’re monitoring delivery performance (e.g., delivery time in hours) across 1000 shipments. Most deliveries take 2–5 hours, but a few take longer due to disruptions (traffic, vendor delay, strikes, etc.).
We want to automatically flag outliers using ML — without manually defining rules.

Uses dataset "Shipment_Delivery_Data.csv"


# Anomaly Detection Demo: Spotting Delivery Disruptions


# Install dependencies (if needed)
#try:
#    import sklearn
#except ImportError:
#    !pip install scikit-learn

#try:
#    import matplotlib
#except ImportError:
#    !pip install matplotlib