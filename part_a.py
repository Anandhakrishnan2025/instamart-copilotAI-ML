"""Part A: audit, clean, store, validate-ready outputs, SQL analysis and pandas cross-check."""
import re, sqlite3
import numpy as np, pandas as pd

import os
for _d in ("data/clean", "outputs", "state", "sql"): os.makedirs(_d, exist_ok=True)
RAW = "data/raw/"
cust = pd.read_csv(RAW + "customers.csv")
prod = pd.read_csv(RAW + "products.csv")
orders = pd.read_csv(RAW + "orders.csv", dtype={"order_amount": str})
# the literal strings "N/A"/"error" are unparseable; read raw strings (no NA parsing) so none is hidden by pandas
raw_amount = pd.read_csv(RAW + "orders.csv", dtype=str, keep_default_na=False)["order_amount"]
non_numeric_amount = int(pd.to_numeric(raw_amount, errors="coerce").isna().sum())

print("=== A2 AUDIT ===")
for name, df in [("customers", cust), ("products", prod), ("orders", orders)]:
    print(f"\n-- {name}: shape={df.shape}")
    print(pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                        "missing_pct": (df.isna().mean() * 100).round(2)}).to_string())
audit = {
    "raw orders rows": len(orders),
    "exact duplicate rows": int(orders.duplicated().sum()),
    "pending rows": int((orders.status == "pending").sum()),
    "distinct order_id": int(orders.order_id.nunique()),
    "non-numeric order_amount": non_numeric_amount,
    "negative delivery_minutes": int((orders.delivery_minutes < 0).sum()),
    "blank delivery_minutes": int(orders.delivery_minutes.isna().sum()),
    "blank age": int(cust.age.isna().sum()),
    "age outside 10-100": int(((cust.age < 10) | (cust.age > 100)).sum()),
    "distinct raw city spellings": int(cust.city.nunique()),
    "distinct non-blank gender spellings": int(cust.gender.dropna().nunique()),
    "blank gender": int(cust.gender.isna().sum()),
}
print("\nAUDIT FIGURES"); [print(f"  {k}: {v}") for k, v in audit.items()]

print("\n=== A3 CLEAN CUSTOMERS ===")
c = cust.copy()
c["city"] = c["city"].str.strip().str.title()
gmap = {"male": "male", "m": "male", "female": "female", "f": "female"}
before_unknown = c.gender.isna().sum()
c["gender"] = c["gender"].fillna("").str.strip().str.lower().map(gmap).fillna("unknown")
bad_age = (c.age < 10) | (c.age > 100)
c.loc[bad_age, "age"] = np.nan
n_missing_age = int(c.age.isna().sum())
c["age"] = c["age"].fillna(c.groupby("city")["age"].transform("median")).round().astype(int)
c["signup_date"] = pd.to_datetime(c["signup_date"])
print("cities:", c.city.value_counts().to_dict())
print("genders:", c.gender.value_counts().to_dict())
print("missing ages after:", int(c.age.isna().sum()), "| age range:", c.age.min(), c.age.max())
print(f"values changed: city standardised {int((cust.city != c.city).sum())}, gender mapped {int((cust.gender.fillna('') != c.gender).sum())}, ages set missing (out of range) {int(bad_age.sum())}, ages imputed {n_missing_age}")

print("\n=== A4 CLEAN ORDERS ===")
o = orders.copy(); log = {}
o["order_ts"] = pd.to_datetime(o["order_ts"]); log["1 order_ts converted"] = len(o)
n = len(o); o = o.drop_duplicates(); log["2 exact duplicates removed"] = n - len(o)
n = len(o); o = o.sort_values(["order_id", "order_ts"]).drop_duplicates("order_id", keep="last"); log["3 older records of same order_id removed"] = n - len(o)
o["order_amount"] = pd.to_numeric(o["order_amount"], errors="coerce")
o = o.merge(prod[["product_id", "category", "price_inr"]], on="product_id", how="left")
miss = o.order_amount.isna()
o.loc[miss, "order_amount"] = (o.price_inr * o.quantity * (1 - o.discount_pct / 100)).round(2)[miss]
log["4 amounts filled from product price"] = int(miss.sum())
neg = o.delivery_minutes < 0; o.loc[neg, "delivery_minutes"] = np.nan; log["5a negative delivery set missing"] = int(neg.sum())
comp = o.status == "completed"
med = o[comp].groupby("category")["delivery_minutes"].transform("median")
fill = comp & o.delivery_minutes.isna()
o.loc[fill, "delivery_minutes"] = med[fill]; log["5b completed delivery filled by category median"] = int(fill.sum())
o = o.drop(columns=["category", "price_inr"]).sort_values("order_id").reset_index(drop=True)
for k, v in log.items(): print(f"  {k}: {v}")
comp = o.status == "completed"
print("rows", len(o), o.status.value_counts().to_dict())
print("missing amount", int(o.order_amount.isna().sum()),
      "| missing delivery completed", int(o[comp].delivery_minutes.isna().sum()),
      "| missing delivery cancelled", int(o[~comp].delivery_minutes.isna().sum()))
print("completed revenue", round(o[comp].order_amount.sum(), 2))

print("\n=== A5 STORE ===")
c.to_csv("data/clean/customers.csv", index=False); prod.to_csv("data/clean/products.csv", index=False)
o_out = o.copy(); o_out["order_ts"] = o_out["order_ts"].dt.strftime("%Y-%m-%d %H:%M:%S")
o_out.to_csv("data/clean/orders.csv", index=False)
con = sqlite3.connect("data/instamart.db")
c_db = c.copy(); c_db["signup_date"] = c_db["signup_date"].dt.strftime("%Y-%m-%d")
c_db.to_sql("customers", con, if_exists="replace", index=False)
prod.to_sql("products", con, if_exists="replace", index=False)
o_out.to_sql("orders", con, if_exists="replace", index=False)
print("tables written:", [r[0] for r in con.execute("select name from sqlite_master where type='table'")])

print("\n=== A6 SQL ===")
qs = [q.strip() for q in re.split(r"^-- Q\d.*$", open("sql/queries.sql").read(), flags=re.M) if q.strip()]
res = [pd.read_sql_query(q, con) for q in qs]
print("Q1 customers:", len(res[0]))
print("Q2 rows:", len(res[1]), "sum:", round(res[1].revenue.sum(), 2)); print("   largest:", res[1].iloc[0].tolist())
print("Q3 products:", len(res[2]))
print("Q4 customers without completed order:", len(res[3]))
print("Q5 customers above average:", len(res[4]), "avg:", res[4].avg_total.iloc[0])
q6 = res[5]; print("Q6 top per city:"); print(q6[q6.row_num == 1][["city", "customer_id", "total_spend"]].to_string(index=False))
print("   rows where RANK != ROW_NUMBER:", int((q6.rnk != q6.row_num).sum()))
print("Q7 daily rows:", len(res[6]), "last cumulative:", res[6].cumulative_revenue.iloc[-1])
con.close()

print("\n=== A7 PANDAS CROSS-CHECK ===")
oc = o[comp].merge(c[["customer_id", "city"]], on="customer_id").merge(prod[["product_id", "category"]], on="product_id")
pv = pd.pivot_table(oc, values="order_amount", index="city", columns="category", aggfunc="sum", margins=True, fill_value=0)
print(pv.round(2).to_string()); print("pivot grand total:", round(pv.loc["All", "All"], 2), "| SQL total:", round(res[1].revenue.sum(), 2))
print(oc.groupby("city").agg(total=("order_amount", "sum"), mean=("order_amount", "mean"), count=("order_amount", "count")).round(2).to_string())
a = o[comp].order_amount
q1, q3 = a.quantile(.25), a.quantile(.75); iqr = q3 - q1
lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"mean {a.mean():.2f} median {a.median():.2f} skew {a.skew():.2f} Q1 {q1:.2f} Q2 {a.median():.2f} Q3 {q3:.2f}")
print(f"fences low {lo:.2f} high {hi:.2f}; outliers below {int((a<lo).sum())} above {int((a>hi).sum())}")
corr = o[comp][["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr()
print(corr.round(3).to_string())
