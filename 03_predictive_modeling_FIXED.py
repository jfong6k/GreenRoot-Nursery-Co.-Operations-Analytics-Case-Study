"""
03 - Predictive modeling: demand forecasting + reorder point optimization
Mirrors 'Collaborate on predictive modeling and optimization projects'.

Two pieces:
  A) Forecast next 30 days of demand per SKU using a seasonal regression
     (day-of-year sinusoidal features + trend), evaluated with a holdout.
  B) Translate the forecast into optimized reorder points (EOQ-style)
     to reduce both stockouts and overstock/carrying cost.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error

plt.style.use("seaborn-v0_8-whitegrid")
DATA = "../data"
CHARTS = "../charts"

orders = pd.read_csv(f"{DATA}/orders.csv", parse_dates=["date"])
inventory = pd.read_csv(f"{DATA}/inventory.csv")

daily = orders.groupby(["date", "category"])["quantity"].sum().reset_index()

# ---------- Feature engineering: seasonal + trend features ----------
def make_features(df):
    doy = df["date"].dt.dayofyear
    t = (df["date"] - df["date"].min()).dt.days
    X = pd.DataFrame({
        "trend": t,
        "sin_annual": np.sin(2 * np.pi * doy / 365),
        "cos_annual": np.cos(2 * np.pi * doy / 365),
        "sin_semiannual": np.sin(4 * np.pi * doy / 365),
        "cos_semiannual": np.cos(4 * np.pi * doy / 365),
        "dow": df["date"].dt.dayofweek,
    })
    X = pd.get_dummies(X, columns=["dow"], drop_first=True)
    return X

results = {}
fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=False)
axes = axes.flatten()

for i, cat in enumerate(sorted(daily["category"].unique())):
    sub = daily[daily.category == cat].sort_values("date").reset_index(drop=True)
    X = make_features(sub)
    y = sub["quantity"].values

    split = int(len(sub) * 0.85)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y[:split], y[split:]

    model = Ridge(alpha=2.0)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mape = np.mean(np.abs((y_test - preds) / np.maximum(y_test, 1))) * 100
    results[cat] = {"mae": mae, "mape": mape, "avg_actual": y_test.mean()}

    ax = axes[i]
    ax.plot(sub["date"].iloc[split:], y_test, label="Actual", color="#2e7d32", alpha=0.8)
    ax.plot(sub["date"].iloc[split:], preds, label="Forecast", color="#c62828", linestyle="--")
    ax.set_title(f"{cat}  (MAE={mae:.1f} units/day, MAPE={mape:.0f}%)", fontsize=10)
    ax.legend(fontsize=8)
    ax.tick_params(axis="x", rotation=30, labelsize=8)

fig.suptitle("Demand Forecast Holdout Performance (last 15% of days), by Category", fontsize=13)
fig.tight_layout()
fig.savefig(f"{CHARTS}/08_forecast_holdout.png", dpi=140)
plt.close(fig)

# ---------- B) Reorder point optimization ----------
# Current reorder points were set with a flat safety-stock heuristic.
# Recompute using demand volatility (std dev) over the trailing 90 days,
# a proper safety-stock formula: SS = z * sigma_d * sqrt(lead_time)
z_score = 1.65  # ~95% service level

recent = orders[orders["date"] >= orders["date"].max() - pd.Timedelta(days=90)]
# IMPORTANT: aggregate to one row per (sku, date) FIRST, summing same-day orders,
# before computing mean/std. Without this step, mean/std are computed over
# individual order-line quantities (~1-3 units each) instead of true daily
# demand totals, which understates both figures by roughly 5x.
daily_sku = recent.groupby(["sku", "date"])["quantity"].sum().reset_index()
sku_stats = daily_sku.groupby("sku")["quantity"].agg(["mean", "std"]).fillna(0)
sku_stats.columns = ["avg_daily_demand", "std_daily_demand"]

opt = inventory.merge(sku_stats, left_on="sku", right_index=True, how="left").fillna(0)
opt["optimized_safety_stock"] = (z_score * opt["std_daily_demand"] * np.sqrt(opt["lead_time_days"])).round(1)
opt["optimized_reorder_point"] = (opt["avg_daily_demand"] * opt["lead_time_days"] + opt["optimized_safety_stock"]).round(1)
opt["current_below_optimized"] = opt["on_hand_units"] < opt["optimized_reorder_point"]
opt["reorder_point_change_pct"] = ((opt["optimized_reorder_point"] - opt["reorder_point"]) / opt["reorder_point"].replace(0, np.nan) * 100).round(1)

opt_out = opt[["sku", "product_name", "category", "on_hand_units", "reorder_point",
               "optimized_reorder_point", "reorder_point_change_pct", "current_below_optimized"]]
opt_out.to_csv(f"{DATA}/reorder_point_optimization.csv", index=False)

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(opt["reorder_point"], opt["optimized_reorder_point"],
           c=opt["current_below_optimized"].map({True: "#c62828", False: "#2e7d32"}), s=45, alpha=0.8)
lims = [0, max(opt["reorder_point"].max(), opt["optimized_reorder_point"].max()) * 1.05]
ax.plot(lims, lims, linestyle="--", color="gray", linewidth=1, label="No change")
ax.set_xlabel("Current Reorder Point (units)")
ax.set_ylabel("Optimized Reorder Point (units)")
ax.set_title("Reorder Point: Heuristic vs. Demand-Volatility-Based Optimization\n(red = currently understocked given optimized point)")
ax.legend()
fig.tight_layout()
fig.savefig(f"{CHARTS}/09_reorder_point_optimization.png", dpi=140)
plt.close(fig)

n_flagged = opt["current_below_optimized"].sum()
avg_change = opt["reorder_point_change_pct"].mean()

print("=== PREDICTIVE MODELING SUMMARY ===")
for cat, r in results.items():
    print(f"  {cat}: MAE={r['mae']:.1f} units/day  MAPE={r['mape']:.0f}%  (avg actual={r['avg_actual']:.1f})")
print(f"\nReorder point optimization: {n_flagged}/{len(opt)} SKUs currently understocked "
      f"relative to volatility-adjusted target.")
print(f"Average reorder point change vs. old heuristic: {avg_change:+.1f}%")
