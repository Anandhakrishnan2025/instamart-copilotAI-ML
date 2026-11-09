# Instamart Customer Intelligence & Support Copilot

## Overview

A single repository that turns dirty order exports into a trusted SQLite database (Part A), segments customers and predicts lapse (Part B), triages support tickets with privacy and cost guardrails (Part C), and answers policy questions and decides refunds with a retrieval-grounded assistant and a tool-calling agent (Part D). All data is synthetic and fictional; all money is in INR. Every model call goes through the supplied offline `mock_llm.py`; no network, key or paid service is used. `generate_data.py` and `mock_llm.py` are unmodified.

| File | Role |
|---|---|
| `part_a.py`, `validate.py`, `sql/queries.sql` | Part A: audit, clean, store, SQL, pandas checks, checkpoints |
| `part_b.py` | Part B: RFM, K-means, PCA, lapse models |
| `part_c.py` | Part C: ticket triage pipeline |
| `rag.py`, `agent.py`, `part_d.py` | Part D: RAG, tools and ReAct agent, runner |

## Reproduce

Run from the repository root (Python 3.11+; developed on 3.12, pandas 3.0.6, scikit-learn 1.9.1).

1. `python -m venv .venv`
2. `source .venv/bin/activate` (Windows: `.venv\Scripts\activate`)
3. `pip install -r requirements.txt`
4. `python generate_data.py` (last line: `generated: customers=1200 products=40 orders_raw=13084 tickets=164`)
5. `python part_a.py`
6. `python validate.py` (prints `ALL CHECKPOINTS PASSED`)
7. `python part_b.py`
8. `python part_c.py`
9. `python part_d.py`

Part B must run before Part C (routing reads `outputs/customer_segments.csv`). Each script prints every number reported below.

## Part A

### A2 Audit of the raw data

Shapes: customers (1200, 5), products (40, 4), orders (13084, 9). Missing values (count / %): customers `age` 61 / 5.08%, `gender` 30 / 2.50%; orders `order_amount` 53 / 0.41% (the literal text `N/A`, which pandas reads as missing), `delivery_minutes` 2541 / 19.42%; every other column 0. Dtypes: ids, city, dates, status, gender are strings; age and delivery_minutes are float64 (they contain blanks); quantity, discount_pct, price_inr are int64. `order_amount` is read as text because it contains `error`/`N/A`.

| Audit figure | Value |
|---|---|
| Raw order rows | 13,084 |
| Exact duplicate rows | 119 |
| Rows with status pending | 656 |
| Distinct order_id | 12,309 |
| Non-numeric order_amount (`N/A` 53 + `error` 67) | 120 |
| Negative delivery_minutes | 106 |
| Blank delivery_minutes | 2,541 |
| Blank age | 61 |
| Age outside 10-100 | 6 |
| Distinct raw city spellings | 24 |
| Distinct non-blank gender spellings | 10 |
| Blank gender | 30 |

(`N/A` is counted from the raw text with `keep_default_na=False`, because pandas would otherwise hide it as a blank.)

### A3 Customer cleaning rules

1. City: strip spaces and title-case. 476 values changed; exactly six cities remain: Pune 227, Delhi 210, Bengaluru 197, Chennai 191, Hyderabad 191, Mumbai 184.
2. Gender: `male/m/Male/MALE` to `male`, `female/f/Female/FEMALE` to `female`, blank to `unknown` (950 values changed). Result: female 610, male 560, unknown 30.
3. Age: 6 values outside 10-100 set to missing; 67 missing ages (61 blank + 6) filled with the city median via `groupby().transform("median")`, rounded to whole years. Result: 0 missing, all ages 18-60.
4. `signup_date` converted to datetime.

### A4 Order cleaning rules (in order)

| # | Rule | Rows/values changed |
|---|---|---|
| 1 | `order_ts` to datetime | 13,084 values |
| 2 | Drop exact duplicate rows | 119 removed |
| 3 | Per `order_id` keep the latest `order_ts` (drops the earlier pending record) | 656 removed |
| 4 | `order_amount` to numeric (unparseable to missing), then fill from `round(price_inr * quantity * (1 - discount_pct/100), 2)` | 112 amounts filled (the other 8 unparseable ones were in rows removed by rules 2-3) |
| 5a | Negative `delivery_minutes` to missing | 105 set missing (106 raw; one sat in a removed duplicate) |
| 5b | Completed orders: fill missing `delivery_minutes` with the median of the product category among completed orders (`groupby().transform`); cancelled orders left missing | 774 filled |

After cleaning: 12,309 rows (one per order_id); 11,110 completed, 1,199 cancelled; 0 pending; 0 missing amounts; 0 missing delivery among completed, 1,199 among cancelled; completed revenue INR 2,998,668.80.

### A5 Storage and validation

Clean tables are in `data/clean/*.csv` and in `data/instamart.db` (tables `customers`, `products`, `orders`; `order_ts` stored as `YYYY-MM-DD HH:MM:SS` text). `validate.py` asserts 14 checkpoints (row counts, status counts, missing values, revenue, cities, genders, ages, tables) and prints `ALL CHECKPOINTS PASSED`.

### A6 SQL results (`sql/queries.sql`, run with `pd.read_sql_query`)

| Query | Result |
|---|---|
| Q1 WHERE + HAVING | 30 customers |
| Q2 three-table join | 30 rows; sum INR 2,998,668.80; largest Bengaluru / Electronics INR 233,909.45 |
| Q3 scalar subquery | 11 products above the average price |
| Q4 LEFT JOIN | 25 customers with no completed order |
| Q5 CTEs | 550 customers above the average total of INR 2,552.06 |
| Q6 window ranks | Top per city: Bengaluru C0082 12,856.20; Chennai C0930 10,151.40; Delhi C0291 13,742.60; Hyderabad C0743 12,730.05; Mumbai C0693 14,766.55; Pune C0426 10,377.10. 8 rows have RANK() different from ROW_NUMBER() |
| Q7 running total | 183 daily rows; last cumulative value INR 2,998,668.80 |

Why ties cause the Q6 difference: tied customers share the same RANK (and the next rank is skipped), while ROW_NUMBER() must still give every row a unique number, so for the second and later rows of a tie the two functions disagree.

### A7 pandas cross-check and exploratory analysis

`pd.pivot_table(..., margins=True)` grand total = INR 2,998,668.80, identical to the SQL Q2 total. Per city (completed orders):

| City | Sum (INR) | Mean (INR) | Count |
|---|---|---|---|
| Bengaluru | 523,643.40 | 276.47 | 1,894 |
| Chennai | 463,629.15 | 272.72 | 1,700 |
| Delhi | 502,829.80 | 243.62 | 2,064 |
| Hyderabad | 528,589.90 | 284.95 | 1,855 |
| Mumbai | 479,058.50 | 297.00 | 1,613 |
| Pune | 500,918.05 | 252.48 | 1,984 |

Completed `order_amount`: mean 269.91, median 135.00, skewness 4.61, Q1 64.80, Q3 297.00, IQR fences -283.50 and 645.30; 0 outliers below, 897 above.

Correlation matrix (completed orders): order_amount/discount_pct -0.095; order_amount/quantity +0.087; order_amount/delivery_minutes -0.003; discount_pct/quantity 0.000; discount_pct/delivery_minutes -0.010; quantity/delivery_minutes -0.013.

**Four questions (under 200 words).** *What is the data about?* 11,110 completed orders from six cities, 40 products in five categories. *Distributions?* order_amount is strongly right-skewed (mean 269.91 vs median 135.00, skewness 4.61); most baskets are small grocery items. *Outliers?* Upper fence 645.30; 897 orders exceed it and none fall below; they are genuine high-priced electronics orders, not errors. *Relationships?* All correlations are weak; the largest in absolute value is order_amount with discount_pct (-0.095), just ahead of order_amount with quantity (+0.087); delivery_minutes is unrelated to the rest. A correlation of this size does not establish causation, and it explains under 1% of the variance.

## Part B

Segments and the lapse model use completed orders only. Segment labels were **not** used as lapse features; **no data dated 2025-09-01 or later entered any lapse feature**; **no encoder, scaler or resampler saw test rows**.

### B1 RFM (snapshot 2025-10-01)

1,175 customers; recency median 19, min 0, max 180; frequency mean 9.46, max 33; monetary mean INR 2,552.06, max INR 14,766.55.

### B2 K-means

| K | Inertia | Silhouette |
|---|---|---|
| 2 | 1820.8 | 0.465 |
| 3 | 1268.2 | 0.445 |
| 4 | 743.1 | 0.526 |
| 5 | 617.4 | 0.480 |
| 6 | 503.2 | 0.482 |
| 7 | 408.5 | 0.416 |
| 8 | 366.5 | 0.412 |

The elbow is at K=4: inertia falls 41% from K=3 to K=4 but only 17% from K=4 to K=5, and the silhouette peaks at K=4 (0.526). Four segments is also a number the retention team can run campaigns for. **Final K = 4.** Inertia at K=8 is always lower than at K=2 because every added centroid can only bring points closer to their nearest centre (inertia reaches 0 when K equals the number of customers), so lower inertia alone rewards over-splitting; we need the elbow, silhouette and interpretability to choose K.

### B3 Segments and PCA

Cluster numbers are only labels and carry no ordering. Premium = the segment with the highest mean monetary value.

| Segment id | Name | Size | Mean recency (days) | Mean frequency | Mean monetary (INR) | Recommended action |
|---|---|---|---|---|---|---|
| 1 | **Premium Big Spenders** (`is_premium`) | 138 | 24.12 | 6.77 | 7,129.70 | Early access, priority support, electronics bundles |
| 2 | Frequent Regulars | 419 | 8.77 | 19.40 | 3,410.02 | Subscriptions, loyalty points, basket-size nudges |
| 0 | Occasional Light Buyers | 436 | 28.86 | 3.77 | 1,057.62 | Free-delivery thresholds, category cross-sell |
| 3 | Lapsing / Dormant | 182 | 109.04 | 2.20 | 685.99 | Win-back coupon, reminder campaign |

PCA(2) on the scaled RFM: explained_variance_ratio_ = 0.634 and 0.218, total 0.852. File: `outputs/customer_segments.csv`.

### B4 Lapse problem

Cutoff 2025-09-01. Cohort: 1,156 customers with a completed order before the cutoff; 443 lapsed (38.32%). Split `train_test_split(test_size=0.2, stratify=y, random_state=42)`: 924 train, 232 test. Features (all from orders before the cutoff plus the customer table): recency, frequency, monetary, average discount, cancellation rate, average delivery minutes, favourite category, city, age, gender, tenure in days. Baseline "never lapsed": accuracy 61.6%, lapsed-class F1 0.

### B5 Preprocessing

A `ColumnTransformer` inside a `Pipeline` applies `StandardScaler` to numeric columns and `OneHotEncoder(drop="first")` (k-1 columns) to city, gender and favourite category; it is fitted on training rows only (inside each CV fold during grid search). Scaler mean of frequency 8.1147 (train only) versus full-cohort mean 8.0285: they differ because the scaler sees only the 924 training rows, a sample of the cohort, which is exactly what keeps the test rows out. Confirmed in code: no feature column has a missing value.

### B6 Tuning and test evaluation

| Model | Grid | Best params | Best CV F1 |
|---|---|---|---|
| LogisticRegression | C in {0.01, 0.1, 1, 10, 100}, cv=5, f1 | C = 100 (edge of grid) | 0.651 |
| RandomForest (100 trees, seed 42) | max_depth {3,5,8} x min_samples_leaf {5,10,20}, cv=5, f1 (45 fits) | depth 8, leaf 10 | 0.687 |

Untouched test set, lapsed class (confusion matrix rows = true, [[TN, FP], [FN, TP]]):

| Model | Confusion | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline | all predicted not lapsed | 0 | 0 | 0 |
| Logistic regression | [[111, 32], [31, 58]] | 0.644 | 0.652 | 0.648 |
| Random forest | [[111, 32], [32, 57]] | 0.640 | 0.640 | 0.640 |

**Selected model: random forest**, chosen on cross-validated F1 only (0.687 vs 0.651) so the test set played no part in the choice; its test F1 is 0.640 (>= 0.55). On test the two models are essentially tied, so the choice rests on the higher CV score and on the forest's ability to capture non-linear spend/frequency patterns. A missed lapsing customer (false negative) costs the whole future revenue of that customer, whereas wrongly targeting a loyal one (false positive) costs only a small coupon or message, so recall deserves more weight than precision, and a threshold below 0.5 would be justified in production.

### B7 Class-balance experiment

Training set only: all 924 original rows kept, 216 duplicated lapsed rows (354 original lapsed rows to 570) drawn with replacement (`random_state=42`) so both classes have 570. Best logistic regression (C=100) refit on it, evaluated on the original test set:

| Model | Precision | Recall | F1 | FP | FN |
|---|---|---|---|---|---|
| LR unbalanced | 0.644 | 0.652 | 0.648 | 32 | 31 |
| LR over-sampled | 0.589 | 0.854 | 0.697 | 53 | 13 |

Recall moved most (+0.20); precision fell by about 0.05 instead of rising. For the retention team this means catching far more would-be lapsers (76 of 89 instead of 58) at the price of contacting 21 more loyal customers, which is a good trade when a campaign is cheap relative to a lost customer.

### B8 Interpretation, persistence, retention list

Top-5 random-forest importances: monetary 0.278 (spend level separates light from heavy buyers), frequency 0.277 (the more often they buy, the less likely they stop), recency 0.111 (time since last order), average delivery minutes 0.062 (slow service), average discount 0.053 (discount-driven buyers).

Logistic-regression coefficients (standardised features): frequency -1.70 (more orders means lower lapse risk, the dominant signal); monetary +0.28 (given the same frequency, big-basket buyers lapse slightly more, partly a collinearity effect with frequency); recency +0.04 (longer silence slightly raises risk; weak because the synthetic orders arrive at random and so carry little recency signal).

`outputs/lapse_model.joblib` was saved with `joblib.dump`, reloaded with `joblib.load`, and reproduces the original test predictions exactly (checked in code). `outputs/retention_list.csv`: of 40 Premium customers in the test set, the 10 with the highest predicted lapse probability were listed (top probability 0.530); **1 of the 10 truly lapsed**. The retention list is illustrative and not a leakage-free evaluation, because the Premium flag comes from the whole-window RFM, which includes the September window that defines the label.

## Part C

### C1-C2 Guardrails and redaction

4 empty tickets (`skipped_empty`) and 4 long tickets (`flagged_long`, estimated 734 tokens > 400) never reach the model. Redaction (email, then phone, then order id, with the exact patterns of the brief): 55 emails, 91 phones, 132 order ids, **278 in total**; re-applying the patterns to the redacted texts finds 0. The user message is exactly the redacted text; an `assert` before every call checks that no pattern matches the prompt.

### C3 Prompts

Zero-shot: 25 estimated tokens. Few-shot: 120 estimated tokens, with an `EXAMPLES:` line and five labelled examples (two fraud). Both contain `CLASSIFY`, the four categories and a JSON-only instruction.

### C4-C5 Retries and validation

Each full run over the 156 eligible tickets had 19 `TransientLLMError`s, all recovered (0 permanent errors). Retry waits fell between 0.0455 and 0.0572 s (range 0.045-0.0575 for retry number 0); one log line per retry is printed and saved in `outputs/retry_log_*.txt`, e.g. `RETRY ticket=T0002 retry_number=0 wait=0.0490s`. Status counts for a full run: **135 classified_json, 12 classified_keyword, 9 flagged_invalid_output** (valid JSON with the out-of-list category `shipping`), plus 4 skipped_empty and 4 flagged_long. Keyword replies are parsed with `re.findall(r"[a-z]+")`.

### C6 Zero-shot versus few-shot

| Run | Classified | Correct | Accuracy | Total tokens | Tokens per classified ticket |
|---|---|---|---|---|---|
| Zero-shot | 147 | 126 | 85.7% | 8,308 | 56.5 |
| Few-shot | 147 | 147 | 100.0% | 23,128 | 157.3 |

Zero-shot confusion (rows = true, columns = predicted):

| true \ pred | billing | delivery | fraud | technical |
|---|---|---|---|---|
| billing | 38 | 0 | 0 | 0 |
| delivery | 0 | 36 | 0 | 0 |
| fraud | 11 | 0 | 23 | 0 |
| technical | 10 | 0 | 0 | 29 |

The few-shot prompt costs about 101 extra tokens per ticket (+14,820 tokens in total, 2.8 times the zero-shot cost) and fixes 21 tickets, i.e. roughly 700 tokens per extra correct ticket. It is worth paying when errors are expensive: here 11 fraud tickets were sent to billing, missing the 2-hour security SLA, so the extra tokens are justified; for low-stakes routing where 85.7% is acceptable, zero-shot saves two thirds of the cost.

### C7 Budget, routing and outputs

`TOKEN_BUDGET = 6000` with the zero-shot prompt (`outputs/triage_budget_demo.csv`): the budget was reached after 6,012 tokens; **43 tickets ended `budget_exhausted`** (99 json + 8 keyword classified, 6 invalid, 4 empty, 4 long). The final `outputs/triage_results.csv` (164 rows) comes from the few-shot run with budget 100,000 and the routing rules of the brief (customers missing from the segment file count as non-premium): security_team 34 (HIGH), premium_desk 11 (HIGH), support_queue 102 (NORMAL), human_review 17 (NORMAL). **Rule-validated by the output checks and routed automatically: 147 tickets** (135 JSON + 12 keyword). **Sent to humans: 17** (4 empty, 4 long, 9 invalid output).

## Part D

### D1-D4 Chunking, retrieval and evaluation

15 chunks (40 words, overlap 10): refund 4, delivery 3, escalation 3, payment 3, privacy 2. TF-IDF + cosine similarity retrieval; for `refund policy late delivery` the top three chunks are all from `refund_policy.txt`. TF (term frequency) measures how often a word occurs in a chunk, IDF (inverse document frequency) down-weights words that appear in many chunks so rare, informative words dominate. A limitation: TF-IDF matches words, not meaning, so a paraphrase such as "money back" for "refund" would not be matched.

| Evaluation | Expected doc in top 3 (of 12) | Expected fact in answer (of 12) | Fallback on unanswerable (of 2) |
|---|---|---|---|
| Overlap 10 | 12 | 12 | 2 |
| Overlap 0 (13 chunks) | 12 | 10 | 2 |

With overlap 0 two questions break: "What delivery time makes a completed order eligible for a full refund?" (the condition "more than 45 minutes" sits at the end of one chunk and the consequence "full refund" in the next, so the answer sentence loses the fact) and "How soon is money returned when a payment fails but is deducted?" (the sentence is cut at a chunk boundary right before "within 48 hours"). Overlap keeps boundary sentences whole in at least one chunk.

### D5-D6 Tools and the seven scenarios

Five tools in `TOOLS_SPEC`/`TOOL_MAP` (`get_order`, `search_policy`, `calculate_refund`, `issue_refund`, `escalate_to_human`); the last two append to `outputs/refund_ledger.jsonl` (4 lines after the full run: S1 and S2 issued, S3 escalated, plus the resumed S1 refund).

| Scenario | Status | Refund (INR) | Model calls | Tokens used |
|---|---|---|---|---|
| S1 | REFUND | 128.25 | 5 | 1,027 |
| S2 | REFUND | 135.00 | 5 | 1,027 |
| S3 | ESCALATE | 1,999.00 | 5 | 1,014 |
| S4 | DENIED | 0.00 | 4 | 709 |
| S5 | DENIED | 0.00 | 3 | 415 |
| S6 | NOT_FOUND | 0.00 | 2 | 124 |
| S7 | FALLBACK_MAX_ITERATIONS | 0.00 | 6 | 3,156 |

Each refund matches `expected_refund_inr` within 0.01. Runs are in `outputs/agent_runs.jsonl` (one line per scenario with every step); the three demonstrations below are in `outputs/agent_demos.jsonl`.

### D7 Stop safely

| Demonstration | Final status |
|---|---|
| S1 with `token_limit=200` | **FALLBACK_TOKEN_LIMIT** (432 tokens after the 3rd call; the requested tool was not executed) |
| S1 with `time_limit_seconds=0` | **FALLBACK_TIMEOUT** (0 model calls) |

S7 ends in FALLBACK_MAX_ITERATIONS under the default `token_limit=6000`. Each fallback returns the steps done so far and a human-handoff summary.

### D8 Crash and resume

Checkpoints are written to `state/<run_id>.json` after every tool observation. A simulated crash was raised after the second tool of S1; `resume_agent(run_id)` continued from the checkpoint. Tools executed across the crash and the resume, in order: **get_order, search_policy, calculate_refund, issue_refund** (four executions, none repeated); final status REFUND, INR 128.25.

### D9 ReAct versus plan-and-execute

My ReAct agent needed 5 model calls for S1 (1,027 tokens), 3 for S5 (415) and 2 for S6 (124). Each call re-sends the growing message history, so tokens grow faster than calls. A plan-and-execute design would ask once for the whole step list and once to summarise: about 2 calls and noticeably fewer tokens (my estimate, not measured) for a normal refund like S1. However, its plan is written before any observation exists. On S6 (order not found) or S5 (cancelled order) the remaining planned steps become invalid; it must detect the surprise, abandon the plan and replan, costing extra calls, whereas ReAct simply reads the observation and answers in 2 or 3 calls. ReAct also lets each decision (refund above INR 1000 means escalate) depend on the actual amount. I would choose ReAct for this refund workflow: money moves, the right next action depends on what the database returns, per-step checkpoints and limits are easy to enforce, and the token overhead (about 1,000 tokens per refund) is small. If cost mattered, a hybrid is possible: a fixed plan for the happy path with ReAct-style replanning only on an error observation.

## Limitations and Next Steps

- All data is synthetic and the LLM is a keyword-matching mock, so the 100% few-shot accuracy and the perfect retrieval say little about real tickets or a real model.
- The Premium flag is a post-hoc whole-window label: the RFM behind it includes the September window that defines the lapse label, so the retention list (1 of 10 truly lapsed) is illustrative and not a leakage-free evaluation. Next step: compute segments as of the cutoff date.
- The lapse model is modest (test F1 about 0.64). Synthetic orders arrive at random, so recency carries little signal; logistic-regression C=100 sits at the edge of the grid and a wider grid, threshold tuning and calibration should be tried.
- Over-sampling raises recall but lowers precision; the right operating point depends on the real campaign cost.
- TF-IDF cannot match paraphrases; dense embeddings and a reranker would help. Chunk size and overlap were fixed by the brief.
- Retry jitter is random, so retry waits differ slightly between runs (the counts do not). Checkpoint resume restarts the wall-clock limit.
- scikit-learn estimator values (K-means, forests) can differ slightly across library versions; `requirements.txt` pins the versions used here.

# instamart-copilotAI-ML
capstone project for AIML
