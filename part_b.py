"""Part B: RFM, K-means segments, PCA, lapse model."""
import numpy as np, pandas as pd, joblib
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import os; os.makedirs("outputs", exist_ok=True)
orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
cust = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
prod = pd.read_csv("data/clean/products.csv")
comp = orders[orders.status == "completed"]

# ---------- B1 RFM ----------
snap = pd.Timestamp("2025-10-01 00:00:00")
g = comp.groupby("customer_id")
rfm = pd.DataFrame({"last": g.order_ts.max(), "frequency": g.order_id.nunique(), "monetary": g.order_amount.sum()})
rfm["recency"] = (snap - rfm["last"]).dt.days
rfm = rfm[["recency", "frequency", "monetary"]].reset_index()
print("=== B1 RFM ===")
print("customers", len(rfm), "| recency median", rfm.recency.median(), "min", rfm.recency.min(), "max", rfm.recency.max())
print("frequency mean %.2f max %d | monetary mean %.2f max %.2f" % (rfm.frequency.mean(), rfm.frequency.max(), rfm.monetary.mean(), rfm.monetary.max()))

# ---------- B2 K-means ----------
X = StandardScaler().fit_transform(rfm[["recency", "frequency", "monetary"]])
print("\n=== B2 K-MEANS ===\nK  inertia  silhouette")
for k in range(2, 9):
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(X)
    print(f"{k}  {km.inertia_:8.1f}  {silhouette_score(X, km.labels_):.3f}")
K = 4
km = KMeans(n_clusters=K, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

# ---------- B3 profile ----------
prof = rfm.groupby("segment_id").agg(size=("customer_id", "count"), recency=("recency", "mean"),
                                     frequency=("frequency", "mean"), monetary=("monetary", "mean")).round(2)
print("\n=== B3 PROFILE (K=4) ===\n", prof.to_string())
pca = PCA(n_components=2).fit(X)
print("PCA explained_variance_ratio_:", pca.explained_variance_ratio_.round(3), "total", pca.explained_variance_ratio_.sum().round(3))
# naming is rule-based on the profile so it is reproducible: premium = highest monetary; then by recency/frequency
prem = prof.monetary.idxmax()
rest = prof.drop(index=prem)
champ = rest.frequency.idxmax()                               # most frequent, recent
rest = rest.drop(index=champ)
dormant = rest.recency.idxmax()                               # longest since last order
names = {int(prem): "Premium Big Spenders", int(champ): "Frequent Regulars", int(dormant): "Lapsing / Dormant"}
for s in prof.index:
    names.setdefault(int(s), "Occasional Light Buyers")
rfm["segment_name"] = rfm.segment_id.map(names)
rfm["is_premium"] = rfm.segment_id == prem
print(names, "| premium segment id:", prem)
rfm.to_csv("outputs/customer_segments.csv", index=False,
           columns=["customer_id", "recency", "frequency", "monetary", "segment_id", "segment_name", "is_premium"])

# ---------- B4 lapse problem ----------
cut = pd.Timestamp("2025-09-01")
pre = orders[orders.order_ts < cut]
pre_c = pre[pre.status == "completed"].merge(prod[["product_id", "category"]], on="product_id")
cohort_ids = pre_c.customer_id.unique()
sep = comp[(comp.order_ts >= cut) & (comp.order_ts < pd.Timestamp("2025-10-01"))]
gg = pre_c.groupby("customer_id")
F = pd.DataFrame({
    "recency": (cut - gg.order_ts.max()).dt.days,
    "frequency": gg.order_id.nunique(),
    "monetary": gg.order_amount.sum(),
    "avg_discount": gg.discount_pct.mean(),
    "avg_delivery": gg.delivery_minutes.mean(),
    "fav_category": gg.category.agg(lambda s: s.value_counts().sort_index().idxmax()),
})
fin = pre[pre.status.isin(["completed", "cancelled"])]
F["cancel_rate"] = fin.groupby("customer_id").status.apply(lambda s: (s == "cancelled").mean()).reindex(F.index).fillna(0)
F = F.reset_index().merge(cust[["customer_id", "city", "age", "gender", "signup_date"]], on="customer_id")
F["tenure_days"] = (cut - F.signup_date).dt.days
F["lapsed"] = (~F.customer_id.isin(sep.customer_id.unique())).astype(int)
F = F.drop(columns="signup_date")
num = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]
cat = ["fav_category", "city", "gender"]
Xf, y = F[num + cat], F["lapsed"]
print("\n=== B4 LAPSE ===\ncohort", len(F), "lapsed", int(y.sum()), f"({y.mean()*100:.2f}%)")
Xtr, Xte, ytr, yte = train_test_split(Xf, y, test_size=0.2, stratify=y, random_state=42)
print("train", len(Xtr), "test", len(Xte))
base = np.zeros(len(yte), dtype=int)
print("baseline accuracy %.3f, lapsed F1 %.3f" % (accuracy_score(yte, base), f1_score(yte, base, zero_division=0)))

# ---------- B5 preprocessing ----------
def prep():
    return ColumnTransformer([("num", StandardScaler(), num), ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), cat)])
pp = prep().fit(Xtr)
print("\n=== B5 ===\nscaler mean frequency (train only): %.4f | full-cohort mean frequency: %.4f" % (pp.named_transformers_["num"].mean_[num.index("frequency")], F.frequency.mean()))
assert not Xf.isna().any().any(); print("no missing values in features: True")

# ---------- B6 tune ----------
lr_pipe = Pipeline([("prep", prep()), ("clf", LogisticRegression(max_iter=1000))])
lr_gs = GridSearchCV(lr_pipe, {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1").fit(Xtr, ytr)
rf_pipe = Pipeline([("prep", prep()), ("clf", RandomForestClassifier(n_estimators=100, random_state=42))])
rf_gs = GridSearchCV(rf_pipe, {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]}, cv=5, scoring="f1").fit(Xtr, ytr)
print("\n=== B6 TUNING ===")
print("LR best", lr_gs.best_params_, "cv F1 %.3f" % lr_gs.best_score_)
print("RF best", rf_gs.best_params_, "cv F1 %.3f" % rf_gs.best_score_)
def ev(name, m):
    p = m.predict(Xte)
    print(f"{name}: confusion [[TN FP],[FN TP]] = {confusion_matrix(yte, p).tolist()} precision {precision_score(yte,p,zero_division=0):.3f} recall {recall_score(yte,p,zero_division=0):.3f} F1 {f1_score(yte,p,zero_division=0):.3f}")
    return p
p_lr = ev("LogReg", lr_gs.best_estimator_); p_rf = ev("RandForest", rf_gs.best_estimator_)
# selection uses cross-validated F1 only (the test set is not used to choose)
best = rf_gs.best_estimator_ if rf_gs.best_score_ >= lr_gs.best_score_ else lr_gs.best_estimator_
best_name = "RandomForest" if best is rf_gs.best_estimator_ else "LogisticRegression"
print("selected model:", best_name, "test F1 %.3f" % f1_score(yte, best.predict(Xte)))

# ---------- B7 class balance ----------
rng = np.random.RandomState(42)
pos = np.where(ytr.values == 1)[0]; neg = np.where(ytr.values == 0)[0]
extra = rng.choice(pos, size=len(neg) - len(pos), replace=True)
idx = np.concatenate([np.arange(len(ytr)), extra])
Xb, yb = Xtr.iloc[idx], ytr.iloc[idx]
print("\n=== B7 OVERSAMPLE ===\nbalanced counts:", yb.value_counts().to_dict())
lr_bal = Pipeline([("prep", prep()), ("clf", LogisticRegression(max_iter=1000, C=lr_gs.best_params_["clf__C"]))]).fit(Xb, yb)
ev("LogReg unbalanced", lr_gs.best_estimator_); ev("LogReg balanced  ", lr_bal)

# ---------- B8 interpret / persist / use ----------
names_out = best_names = rf_gs.best_estimator_.named_steps["prep"].get_feature_names_out()
imp = pd.Series(rf_gs.best_estimator_.named_steps["clf"].feature_importances_, index=names_out).sort_values(ascending=False)
print("\n=== B8 ===\nTop-5 RF importances:\n", imp.head(5).round(4).to_string())
coef = pd.Series(lr_gs.best_estimator_.named_steps["clf"].coef_[0], index=lr_gs.best_estimator_.named_steps["prep"].get_feature_names_out())
print("LR coefficients (selected):\n", coef[["num__recency", "num__frequency", "num__monetary", "num__avg_delivery", "num__cancel_rate"]].round(3).to_string())
joblib.dump(best, "outputs/lapse_model.joblib")
re = joblib.load("outputs/lapse_model.joblib")
print("reloaded predictions identical:", bool((re.predict(Xte) == best.predict(Xte)).all()))
prem_ids = set(rfm.loc[rfm.is_premium, "customer_id"])
te = F.loc[Xte.index, ["customer_id"]].copy()
te["lapse_probability"] = best.predict_proba(Xte)[:, 1]; te["actually_lapsed"] = yte.values
ret = te[te.customer_id.isin(prem_ids)].sort_values("lapse_probability", ascending=False).head(10)
ret.to_csv("outputs/retention_list.csv", index=False)
print("retention list rows", len(ret), "| truly lapsed", int(ret.actually_lapsed.sum()), "| premium customers in test:", int(te.customer_id.isin(prem_ids).sum()))
print(ret.round(3).to_string(index=False))
