"""Validation of Part A reference checkpoints."""
import sqlite3, pandas as pd
o = pd.read_csv("data/clean/orders.csv"); c = pd.read_csv("data/clean/customers.csv")
db = sqlite3.connect("data/instamart.db")
comp = o[o.status == "completed"]
checks = {
    "12,309 order rows": len(o) == 12309,
    "unique order_id": o.order_id.is_unique,
    "11,110 completed": len(comp) == 11110,
    "1,199 cancelled": (o.status == "cancelled").sum() == 1199,
    "no pending": (o.status == "pending").sum() == 0,
    "no missing amount": o.order_amount.isna().sum() == 0,
    "no missing delivery (completed)": comp.delivery_minutes.isna().sum() == 0,
    "1,199 missing delivery (cancelled)": o[o.status == "cancelled"].delivery_minutes.isna().sum() == 1199,
    "completed revenue 2,998,668.80": round(comp.order_amount.sum(), 2) == 2998668.80,
    "six cities": c.city.nunique() == 6 and c.city.value_counts().to_dict() == {"Pune": 227, "Delhi": 210, "Bengaluru": 197, "Chennai": 191, "Hyderabad": 191, "Mumbai": 184},
    "genders 610/560/30": c.gender.value_counts().to_dict() == {"female": 610, "male": 560, "unknown": 30},
    "ages complete 18-60": c.age.notna().all() and c.age.between(18, 60).all(),
    "db tables": {r[0] for r in db.execute("select name from sqlite_master where type='table'")} == {"customers", "products", "orders"},
    "db order count": db.execute("select count(*) from orders").fetchone()[0] == 12309,
}
for k, v in checks.items():
    assert v, f"FAILED: {k}"
print("ALL CHECKPOINTS PASSED")
