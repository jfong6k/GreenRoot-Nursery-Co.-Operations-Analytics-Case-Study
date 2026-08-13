# GreenRoot Nursery Co. — Operations Analytics Case Study

*A portfolio project built to mirror the Data Analyst, Operations Analytics role
at Fast Growing Trees: reporting, ad-hoc analysis, and predictive modeling for
an online plant/tree retailer.*

## Why this project

The role asks for someone to support merchandising, inventory, and customer
support with recurring reports, one-off analyses, and predictive/optimization
models. Rather than a generic Titanic/Iris notebook, this project simulates
the actual shape of that business: seasonal plant demand, multi-region
shipping, supplier lead times, and the link between delivery performance and
customer reviews — then answers three questions an Ops Analytics team would
realistically be asked.

Since Fast Growing Trees' real data isn't public, I generated an 18-month,
148K-order synthetic dataset (`data/generate_data.py`) with realistic
mechanics: spring/fall planting-season demand spikes, category-level
seasonality, regional shipping delays tied to supplier location, and reviews
that degrade after late deliveries. The rest of the project treats that data
as if it were real.

## Project structure

```
plant-nursery-analytics/
├── data/
│   ├── generate_data.py            # synthetic data generator
│   ├── orders.csv                  # ~148K order-line records
│   ├── inventory.csv               # current stock snapshot, 61 SKUs
│   ├── skus.csv                    # product catalog
│   └── reorder_point_optimization.csv  # output of script 03
├── notebooks/
│   ├── 01_reporting_dashboard.py   # recurring KPI reporting
│   ├── 02_adhoc_analysis.py        # one-off stakeholder question
│   └── 03_predictive_modeling.py   # forecasting + reorder point optimization
└── charts/                         # all output visualizations (.png)
```

Colab versions of all three notebooks are also included, with upload/unzip
steps and absolute paths adapted for running outside a local folder
structure.

## 1. Reporting — Ops Health Dashboard
*(maps to "Build, maintain and enhance reporting")*

A recurring report a merchandising/inventory lead would check weekly:

- **Revenue & order volume trend** — strong, predictable seasonality with
  two peaks a year. Spring (March–April) is the dominant peak, more than
  doubling weekly revenue from an ~$85–100K baseline up to ~$220–230K/week
  in both 2024 and 2025. A smaller fall peak (Sept–Oct, ~$140K/week) is
  roughly 60% the size of spring. Revenue and order volume track each other
  almost exactly, suggesting swings are driven by order volume rather than
  average order value.
- **On-time delivery rate by category** — every category misses the 95% SLA
  target, clustered tightly in a 90.9–92.5% range. The narrow spread across
  categories points to a structural/logistics issue rather than a
  product-specific one (e.g. packaging fragility).
- **Inventory health** — roughly 40% of SKUs (24–25 of 61) were below their
  reorder point at the snapshot date. Flowering Shrubs stood out as the
  highest-risk category, with the most SKUs flagged red — consistent with
  it also having the worst on-time delivery rate.
- **Revenue mix by category** — fairly diversified, no category above ~23%
  of revenue. Shade Trees lead in revenue despite being lower-volume, due
  to higher unit price; Succulents are the smallest slice (~8%) despite
  high volume, due to low unit price.

**Key output:** `charts/01_weekly_revenue_trend.png`, `02_otd_by_category.png`,
`03_inventory_health.png`, `04_revenue_mix.png`

## 2. Ad-hoc analysis — "Is late shipping hurting West Coast reviews?"
*(maps to "Execute ad-hoc data analyses")*

**The question (as it would come from merchandising/CS):**
*"We think late shipments are dragging down reviews, especially on the West
Coast. Can you confirm and size it?"*

**Findings:**
- Late deliveries are associated with a large, statistically significant
  drop in review score: **4.50 → 3.35 average** (t = 119.4, p ≈ 0.0) —
  confirmed, and the effect size is substantial (more than a full star).
- West Coast (CA/WA/AZ) on-time rate is 90.6% vs. 92.1% elsewhere — a real
  but modest overall gap. The review-score gap by region alone is small
  (4.41 vs. 4.38), because only ~9–10% of West Coast orders actually ship
  late — the regional dip is a diluted version of the much sharper
  on-time/late effect above, not a separate phenomenon.
- **Root cause:** narrowing to supplier region for West Coast orders only,
  Southeast-sourced SKUs have an 85.4% on-time rate — clearly the worst,
  roughly 6–7 points behind Pacific NW (92.6%), Texas (91.9%), and Midwest
  (91.8%). This isolates the problem to a specific shipping lane
  (Southeast → West Coast), not general West Coast capacity.
- West Coast orders shipped late are far more likely to leave a 1–2 star
  review than West Coast orders shipped on time, and a meaningful share of
  18-month West Coast revenue is exposed to this gap.

**Recommendation:** route West Coast orders for Southeast-sourced SKUs
through a Pacific NW or Texas fulfillment point where possible, or flag
those SKU/region combinations for a regional stocking pilot.

**Key output:** `charts/05_review_vs_ontime.png`, `06_west_coast_deep_dive.png`,
`07_root_cause_supplier_region.png`

## 3. Predictive modeling — demand forecasting & reorder point optimization
*(maps to "Collaborate on predictive modeling and optimization projects")*

**A) Demand forecasting.** A Ridge regression per category using seasonal
(sin/cos annual + semiannual), trend, and day-of-week features, evaluated on
a holdout of the most recent ~15% of days. The model captures the overall
seasonal *decline* (spring peak tapering into summer) well across all six
categories, but is consistently smoother than actual day-to-day demand.
MAPE ranges 11% (Houseplants) to 26% (Flowering Shrubs) — Flowering Shrubs
is both the most volatile and, consistent with the reporting/ad-hoc
findings above, the most operationally troublesome category overall. This
is deliberately a simple, explainable model — appropriate for a first pass
a business stakeholder can trust and that's cheap to retrain weekly, rather
than a black-box model that's hard to justify to merchandising.

**B) Reorder point optimization.** The existing reorder points were set with
a flat safety-stock heuristic. This recomputes them using each SKU's actual
demand volatility over the trailing 90 days:

```
Safety Stock = z × σ_daily_demand × √(lead_time_days)
Reorder Point = avg_daily_demand × lead_time_days + Safety Stock
```

with `z = 1.65` (~95% service level). The corrected result: **28 of 61 SKUs
(46%) are currently understocked** relative to the volatility-adjusted
target, with an average reorder point change of **+9.1%** versus the old
heuristic. Points cluster around the "no change" line across the full
range rather than skewing in one direction — the old heuristic wasn't
systematically wrong, just noisy at the individual-SKU level. This points
toward a SKU-by-SKU replenishment review rather than a blanket policy
change: prioritize the flagged SKUs, especially in high-volatility
categories like Flowering Shrubs, while leaving stable high-volume SKUs
largely as-is.

**Key output:** `charts/08_forecast_holdout.png`, `09_reorder_point_optimization.png`,
`data/reorder_point_optimization.csv`

## A bug I caught and fixed along the way

The first version of the reorder point optimization step produced a result
that looked too good to be true: *every single SKU's* optimized reorder
point came in ~80% lower than the existing one, with zero SKUs flagged as
understocked. That uniformity was the tell — a real result should be messy,
not one-directional across all 61 SKUs.

**The cause:** each row in `orders.csv` is one order *line* (typically 1–3
units), not one day's total demand. The buggy code computed the mean/std of
`quantity` directly, grouped only by `sku` — which measures average *order
size*, not average *daily demand*. On days with multiple orders for the
same SKU, those orders were never summed together first, so daily demand
was understated by roughly 5x.

**The fix:** aggregate to one row per `(sku, date)` — summing same-day
orders — *before* computing the mean/std used in the safety-stock formula:

```python
daily_sku = recent.groupby(["sku", "date"])["quantity"].sum().reset_index()
sku_stats = daily_sku.groupby("sku")["quantity"].agg(["mean", "std"]).fillna(0)
```

After the fix, the numbers above (46% of SKUs flagged, +9.1% average
change) look like a normal, mixed result rather than a suspiciously uniform
one — which is itself a decent signal the fix was correct.

## How I'd talk about this in an interview

- Why synthetic data, and how I made it realistic enough to be useful
  (seasonality tied to actual planting seasons, delay probability tied to
  supplier-region/destination distance, reviews degrading after late
  shipments — not just random noise).
- Why I chose a simple, explainable Ridge regression over something fancier
  for the forecast — in an ops analytics context, a model the business can
  trust and retrain cheaply usually beats a marginal accuracy gain.
- How the three pieces connect: the reporting dashboard *surfaces* the SLA
  miss, the ad-hoc analysis *explains* it and sizes the business impact, and
  the predictive model *proposes a fix* (better reorder points, and
  implicitly, better regional sourcing).
- The bug story above — a genuinely useful answer to "tell me about a time
  you caught a mistake in your own analysis." A result that's too clean or
  too uniform is usually a sign to double-check the code, not a reason to
  trust it more.

## Running it yourself

**Locally:**
```bash
cd data && python3 generate_data.py
cd ../notebooks
python3 01_reporting_dashboard.py
python3 02_adhoc_analysis.py
python3 03_predictive_modeling.py
```

**In Colab:** open any of the `*_colab.ipynb` notebooks, run the upload
cell, select `plant-nursery-analytics.zip` when prompted, then run the
remaining cells top to bottom.

Requires: `pandas`, `numpy`, `matplotlib`, `scikit-learn`, `scipy`.
