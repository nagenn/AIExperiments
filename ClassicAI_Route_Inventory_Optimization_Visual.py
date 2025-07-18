
# Classic AI Demo: Route & Inventory Optimization with Visualization
# Works in both Google Colab and VS Code

# Install dependencies (for Google Colab)
#try:
#    import ortools
#except ImportError:
#    !pip install ortools

#try:
#    import matplotlib
#except ImportError:
#    !pip install matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
from scipy.optimize import linprog

# Load data
inventory_df = pd.read_csv("Store_Inventory.csv")
distance_matrix_df = pd.read_csv("Store_Distance_Matrix.csv", index_col=0)

# ----------- ROUTE OPTIMIZATION WITH VISUALIZATION -----------

print("=== ROUTE OPTIMIZATION ===")

# Simulated store coordinates (for visual only)
coords = {
    'Store A': (2, 3),
    'Store B': (8, 1),
    'Store C': (6, 6),
    'Store D': (1, 8),
    'Store E': (7, 9),
    'Store F': (4, 4)
}

# Convert distance matrix to numpy array
distance_matrix = distance_matrix_df.values

def create_data_model():
    return {
        'distance_matrix': distance_matrix.tolist(),
        'num_vehicles': 1,
        'depot': 0
    }

def print_solution(manager, routing, solution):
    index = routing.Start(0)
    route = []
    while not routing.IsEnd(index):
        route.append(manager.IndexToNode(index))
        index = solution.Value(routing.NextVar(index))
    route.append(manager.IndexToNode(index))
    return route

data = create_data_model()
manager = pywrapcp.RoutingIndexManager(len(data['distance_matrix']),
                                       data['num_vehicles'], data['depot'])
routing = pywrapcp.RoutingModel(manager)

def distance_callback(from_index, to_index):
    return data['distance_matrix'][manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

transit_callback_index = routing.RegisterTransitCallback(distance_callback)
routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

search_parameters = pywrapcp.DefaultRoutingSearchParameters()
search_parameters.first_solution_strategy = (
    routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC)

solution = routing.SolveWithParameters(search_parameters)

if solution:
    route_indices = print_solution(manager, routing, solution)
    store_names = distance_matrix_df.index.tolist()
    route_names = [store_names[i] for i in route_indices]
    print("Optimal Route:")
    print(" ➡ ".join(route_names))

    # Visualize the route
    plt.figure(figsize=(8, 6))
    for store, (x, y) in coords.items():
        plt.plot(x, y, 'o')
        plt.text(x+0.1, y+0.1, store, fontsize=12)

    route_coords = [coords[store_names[i]] for i in route_indices]
    xs, ys = zip(*route_coords)
    plt.plot(xs, ys, '-o', color='blue')
    plt.title("Optimal Delivery Route")
    plt.grid(True)
    plt.show()
else:
    print("No solution found.")

# ----------- INVENTORY OPTIMIZATION WITH TABLE OUTPUT -----------

print("\n=== INVENTORY BALANCING ===")

# Inventory thresholds
min_threshold = 50
max_threshold = 150
sku = 'Milk'
store_names = inventory_df['Store'].tolist()
stock = inventory_df[sku].values

surplus_indices = [i for i, s in enumerate(stock) if s > max_threshold]
deficit_indices = [i for i, s in enumerate(stock) if s < min_threshold]

if not surplus_indices or not deficit_indices:
    print("No balancing needed for", sku)
else:
    n_vars = len(surplus_indices) * len(deficit_indices)
    c = []
    A_eq = np.zeros((len(surplus_indices) + len(deficit_indices), n_vars))
    b_eq = []
    bounds = []

    var_idx = 0
    for i, s_idx in enumerate(surplus_indices):
        surplus_amt = stock[s_idx] - max_threshold
        b_eq.append(surplus_amt)
        for j, d_idx in enumerate(deficit_indices):
            A_eq[i, var_idx] = 1
            var_idx += 1

    for j, d_idx in enumerate(deficit_indices):
        deficit_amt = min_threshold - stock[d_idx]
        b_eq.append(deficit_amt)

    var_idx = 0
    for i, s_idx in enumerate(surplus_indices):
        for j, d_idx in enumerate(deficit_indices):
            A_eq[len(surplus_indices) + j, var_idx] = 1
            cost = distance_matrix[s_idx][d_idx]
            c.append(cost)
            bounds.append((0, None))
            var_idx += 1

    res = linprog(c, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

    if res.success:
        print(f"Balancing Plan for {sku}:")
        x = res.x
        transfers = []
        var_idx = 0
        for i, s_idx in enumerate(surplus_indices):
            for j, d_idx in enumerate(deficit_indices):
                qty = x[var_idx]
                if qty > 1e-2:
                    transfers.append({
                        'From': store_names[s_idx],
                        'To': store_names[d_idx],
                        'Quantity': round(qty, 1)
                    })
                var_idx += 1
        if transfers:
            df = pd.DataFrame(transfers)
            display(df)
        else:
            print("No significant transfers needed.")
    else:
        print("No feasible solution found for inventory balancing.")
