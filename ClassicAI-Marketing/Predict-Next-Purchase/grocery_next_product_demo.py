import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
import itertools

CSV_PATH = "grocery_transactions.csv"

# --- Build co-occurrence counts ---
def cooccurrence_fallback_scores(customer_baskets):
    transactions = customer_baskets["ProductList"].tolist()
    item_counts = Counter()
    for t in transactions:
        item_counts.update(set(t))
    pair_counts = Counter()
    for t in transactions:
        items = sorted(set(t))
        for a, b in itertools.combinations(items, 2):
            pair_counts[(a, b)] += 1
    return item_counts, pair_counts

def conditional_score(history, candidates, item_counts, pair_counts):
    scores = defaultdict(float)
    hist_list = sorted(history)
    for j in candidates:
        s = 0.0
        for i in hist_list:
            if i == j:
                continue
            a, b = sorted((i, j))
            co = pair_counts.get((a, b), 0)
            cnt_i = item_counts.get(i, 1)
            s += co / cnt_i  # approximate P(j|i)
        scores[j] = s
    return scores

def predict_next_products(customer_id, top_k=3, plot=True):
    # Load raw dataset
    df = pd.read_csv(CSV_PATH)
    df.columns = df.columns.str.strip()

    if customer_id not in set(df["CustomerID"].values):
        print("Customer ID not found.")
        return

    # --- RAW PURCHASE HISTORY ---
    cols = [c for c in ["CustomerID", "PurchaseDate", "ProductSequence"] if c in df.columns]
    raw_history = df[df["CustomerID"] == customer_id][cols]
    print(f"All purchases for Customer {customer_id}:\n{raw_history.to_string(index=False)}\n")

    # --- AGGREGATED HISTORY ---
    history = set()
    for seq in df[df["CustomerID"] == customer_id]["ProductSequence"]:
        history.update(seq.split(", "))
    print(f"Aggregated Purchase History (unique items): {sorted(history)}\n")

    # --- CUSTOMER-LEVEL BASKETS ---
    customer_baskets = (
        df.groupby("CustomerID")["ProductSequence"]
          .apply(lambda x: ", ".join(x))
          .reset_index()
    )
    customer_baskets["ProductList"] = customer_baskets["ProductSequence"].apply(
        lambda x: list(dict.fromkeys(x.split(", ")))
    )

    # --- ALWAYS USE CO-OCCURRENCE ---
    item_counts, pair_counts = cooccurrence_fallback_scores(customer_baskets)
    all_items = set(item_counts.keys())
    candidates = sorted(all_items - history)
    scores = conditional_score(history, candidates, item_counts, pair_counts)

    max_s = max(scores.values()) if scores else 1.0
    preds = sorted(
        [(j, (scores[j]/max_s if max_s else 0)) for j in candidates],
        key=lambda x: x[1], reverse=True
    )[:top_k]

    # --- SHOW TOP 3 PREDICTIONS ---
    print("Top 3 Predicted Next Products (ranked):")
    for rank, (item, conf) in enumerate(preds, start=1):
        print(f"{rank}. {item} | Score: {conf:.2f}")

    # --- POPUP CHART ---
    if plot:
        items = [p[0] for p in preds]
        scores = [p[1] for p in preds]
        plt.figure(figsize=(6, 4))
        plt.bar(items, scores, color="skyblue")
        plt.xlabel("Predicted Products")
        plt.ylabel("Score")
        plt.title(f"Top 3 Predictions for Customer {customer_id}")
        plt.tight_layout()
        plt.show(block=True)

if __name__ == "__main__":
    try:
        cid = int(input("Enter Customer ID (1–50): ").strip())
        predict_next_products(cid, top_k=3, plot=True)
    except ValueError:
        print("Please enter a valid numeric Customer ID.")
