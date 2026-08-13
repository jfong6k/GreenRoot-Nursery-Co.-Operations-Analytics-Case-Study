"""
Synthetic data generator for GreenRoot Nursery Co.
Mimics the kind of operational data an online plant/tree retailer
(e.g. Fast Growing Trees) would have: orders, inventory, and shipping.

Run: python3 generate_data.py
Outputs: orders.csv, inventory.csv, skus.csv in this folder
"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

# ---------------------------------------------------------------
# 1. SKU catalog (plants/trees) with seasonality profiles
# ---------------------------------------------------------------
categories = {
    "Fruit Trees": {"base_demand": 45, "spring_boost": 2.5, "price": (35, 120)},
    "Shade Trees": {"base_demand": 30, "spring_boost": 2.0, "price": (60, 250)},
    "Flowering Shrubs": {"base_demand": 55, "spring_boost": 3.0, "price": (25, 80)},
    "Privacy Hedges": {"base_demand": 40, "spring_boost": 1.8, "price": (30, 150)},
    "Houseplants": {"base_demand": 70, "spring_boost": 1.2, "price": (15, 60)},
    "Succulents": {"base_demand": 65, "spring_boost": 1.1, "price": (8, 30)},
}

skus = []
sku_id = 1000
for cat, props in categories.items():
    n_skus = np.random.randint(8, 14)
    for i in range(n_skus):
        skus.append({
            "sku": f"SKU{sku_id}",
            "product_name": f"{cat[:-1] if cat.endswith('s') else cat} Variety {i+1}",
            "category": cat,
            "unit_cost": round(np.random.uniform(*props["price"]) * 0.45, 2),
            "unit_price": None,  # set below
            "base_daily_demand": props["base_demand"] / n_skus,
            "spring_boost": props["spring_boost"],
            "lead_time_days": np.random.choice([7, 10, 14, 21], p=[0.3, 0.35, 0.25, 0.1]),
            "supplier_region": np.random.choice(["Southeast", "Pacific NW", "Midwest", "Texas"]),
        })
        sku_id += 1

skus_df = pd.DataFrame(skus)
skus_df["unit_price"] = (skus_df["unit_cost"] / 0.45).round(2)  # ~55% gross margin
skus_df.to_csv("skus.csv", index=False)

# ---------------------------------------------------------------
# 2. Daily orders over 18 months, with seasonality + noise
# ---------------------------------------------------------------
start_date = datetime(2024, 1, 1)
n_days = 545  # ~18 months
dates = [start_date + timedelta(days=i) for i in range(n_days)]

states = ["CA", "TX", "FL", "NY", "GA", "OH", "PA", "NC", "MI", "IL", "AZ", "WA"]
state_weights = [0.14, 0.11, 0.10, 0.09, 0.07, 0.06, 0.06, 0.06, 0.05, 0.05, 0.05, 0.04]
state_weights = np.array(state_weights) / sum(state_weights)

orders = []
order_id = 500000

for date in dates:
    doy = date.timetuple().tm_yday
    # spring/fall planting season peaks (~day 90 and ~day 270)
    season_mult = 1 + 0.9 * np.exp(-((doy - 100) ** 2) / (2 * 35 ** 2)) \
                    + 0.5 * np.exp(-((doy - 270) ** 2) / (2 * 30 ** 2))
    dow_mult = 1.15 if date.weekday() in (5, 6) else 1.0
    # occasional promo spikes
    promo_mult = 1.6 if np.random.random() < 0.03 else 1.0

    for _, row in skus_df.iterrows():
        expected = row["base_daily_demand"] * season_mult * dow_mult * promo_mult
        if row["spring_boost"] > 2:
            expected *= (1 + 0.6 * np.exp(-((doy - 95) ** 2) / (2 * 25 ** 2)))
        n_units = np.random.poisson(max(expected, 0.05))
        if n_units == 0:
            continue

        n_orders_for_sku = max(1, int(n_units * np.random.uniform(0.5, 0.8)))
        remaining = n_units
        for _ in range(n_orders_for_sku):
            qty = min(remaining, np.random.choice([1, 1, 1, 2, 2, 3], p=[0.5, 0.15, 0.15, 0.1, 0.05, 0.05]))
            remaining -= qty
            if qty <= 0:
                continue
            state = np.random.choice(states, p=state_weights)
            promised_days = np.random.choice([3, 4, 5], p=[0.5, 0.35, 0.15])
            # ship delay risk higher during peak season (capacity strain) and for far states
            distance_penalty = 1 if state in ("CA", "WA", "AZ") and row["supplier_region"] == "Southeast" else 0
            delay_prob = 0.06 + 0.10 * (season_mult > 1.5) + 0.08 * distance_penalty
            actual_days = promised_days + (np.random.choice([0, 1, 2, 3, 5]) if np.random.random() < delay_prob else 0)
            on_time = actual_days <= promised_days

            # review score influenced by on-time delivery + baseline product quality noise
            base_review = np.random.normal(4.5, 0.4)
            if not on_time:
                base_review -= np.random.uniform(0.5, 1.8)
            review_score = float(np.clip(round(base_review), 1, 5))
            leaves_review = np.random.random() < 0.35

            orders.append({
                "order_id": order_id,
                "date": date.strftime("%Y-%m-%d"),
                "sku": row["sku"],
                "category": row["category"],
                "quantity": int(qty),
                "unit_price": row["unit_price"],
                "customer_state": state,
                "promised_ship_days": promised_days,
                "actual_ship_days": int(actual_days),
                "on_time": bool(on_time),
                "review_score": review_score if leaves_review else np.nan,
            })
            order_id += 1
            if remaining <= 0:
                break

orders_df = pd.DataFrame(orders)
orders_df.to_csv("orders.csv", index=False)

# ---------------------------------------------------------------
# 3. Inventory snapshot (current stock levels vs. reorder points)
# ---------------------------------------------------------------
recent = orders_df[orders_df["date"] >= (dates[-1] - timedelta(days=60)).strftime("%Y-%m-%d")]
avg_daily_demand = recent.groupby("sku")["quantity"].sum() / 60

inventory = []
for _, row in skus_df.iterrows():
    add = avg_daily_demand.get(row["sku"], 0.5)
    safety_stock = round(add * row["lead_time_days"] * np.random.uniform(0.3, 0.6), 1)
    reorder_point = round(add * row["lead_time_days"] + safety_stock, 1)
    # simulate current on-hand stock — some below reorder point (stockout risk)
    on_hand = max(0, round(reorder_point * np.random.uniform(0.3, 1.8)))
    inventory.append({
        "sku": row["sku"],
        "product_name": row["product_name"],
        "category": row["category"],
        "avg_daily_demand_60d": round(add, 2),
        "lead_time_days": row["lead_time_days"],
        "safety_stock": safety_stock,
        "reorder_point": reorder_point,
        "on_hand_units": on_hand,
        "supplier_region": row["supplier_region"],
    })

inventory_df = pd.DataFrame(inventory)
inventory_df.to_csv("inventory.csv", index=False)

print(f"Generated {len(orders_df):,} orders across {len(skus_df)} SKUs and {n_days} days.")
print(f"Inventory snapshot: {len(inventory_df)} SKUs, "
      f"{(inventory_df.on_hand_units < inventory_df.reorder_point).sum()} below reorder point.")
