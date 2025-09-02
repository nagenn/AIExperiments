import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, association_rules

# Load dataset
df = pd.read_csv("grocery_transactions.csv")

# Convert 'ProductSequence' column into lists
transactions = df['ProductSequence'].apply(lambda x: x.split(", ")).tolist()

# Encode transactions
te = TransactionEncoder()
te_data = te.fit(transactions).transform(transactions)
df_encoded = pd.DataFrame(te_data, columns=te.columns_)

# Frequent itemsets & association rules
frequent_itemsets = apriori(df_encoded, min_support=0.1, use_colnames=True)
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)

# Function to predict next products for a given customer
def predict_next_products(customer_id):
    customer_basket = df[df['CustomerID'] == customer_id]['ProductSequence'].values[0].split(", ")
    print(f"Customer {customer_id} Basket: {customer_basket}\n")
    
    # Filter rules where antecedents are subset of customer's basket
    possible_rules = rules[rules['antecedents'].apply(lambda x: x.issubset(customer_basket))]
    
    if possible_rules.empty:
        print("No strong prediction could be made for this basket.")
        return
    
    # Show predictions
    print("Predicted Next Product(s):")
    for _, row in possible_rules.iterrows():
        next_product = list(row['consequents'])
        print(f"- {next_product} | Confidence: {row['confidence']:.2f} | Lift: {row['lift']:.2f}")

# Ask user for input
try:
    customer_id = int(input("Enter Customer ID (1–50): "))
    if customer_id in df['CustomerID'].values:
        predict_next_products(customer_id)
    else:
        print("Invalid Customer ID. Please try again.")
except ValueError:
    print("Please enter a valid numeric Customer ID.")
