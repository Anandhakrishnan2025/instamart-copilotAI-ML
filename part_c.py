"""Part C: support-ticket triage with guardrails (offline mock LLM)."""
import json, random, re, time
from collections import Counter
import pandas as pd
import mock_llm
from mock_llm import est_tokens, TransientLLMError

import os; os.makedirs("outputs", exist_ok=True)
MAX_INPUT_TOKENS, BASE_DELAY, MAX_RETRIES = 400, 0.05, 3
CATEGORIES = ["billing", "delivery", "technical", "fraud"]
random.seed(7)  # only affects retry jitter

# ---- redaction (exact patterns and order from the brief) ----
PATTERNS = [("email", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
            ("phone", re.compile(r"\b[6-9]\d{9}\b"), "[PHONE]"),
            ("order_id", re.compile(r"ORD-\d{6}"), "[ORDER_ID]")]
def redact(text, counter=None):
    for name, pat, ph in PATTERNS:
        text, n = pat.subn(ph, text)
        if counter is not None: counter[name] += n
    return text

# ---- prompts ----
ZERO_SHOT = ('You CLASSIFY support tickets into one of: billing, delivery, technical, fraud. '
             'Reply with JSON only: {"category": "...", "reason": "..."}')
FEW_SHOT = ZERO_SHOT + (
    '\nEXAMPLES:\n'
    'Ticket: I was charged twice for [ORDER_ID]. -> {"category": "billing", "reason": "double charge"}\n'
    'Ticket: The rider never delivered [ORDER_ID]. -> {"category": "delivery", "reason": "order not received"}\n'
    'Ticket: The app freezes when I apply a coupon. -> {"category": "technical", "reason": "app error"}\n'
    'Ticket: Someone hacked my account and used my wallet. -> {"category": "fraud", "reason": "unauthorized access"}\n'
    'Ticket: There is an unauthorized transaction I did not make. -> {"category": "fraud", "reason": "unauthorized transaction"}')

def call_with_retry(messages, ticket_id, stats, log):
    for n in range(MAX_RETRIES + 1):
        try:
            return mock_llm.chat(messages)
        except TransientLLMError as e:
            stats["transient_errors"] += 1
            if n == MAX_RETRIES: raise
            wait = BASE_DELAY * 2 ** n * random.uniform(0.9, 1.15)
            log.append(f"RETRY ticket={ticket_id} retry_number={n} wait={wait:.4f}s")
            time.sleep(wait)

def parse(content):
    """Returns (status, category). Never guesses."""
    try:
        obj = json.loads(content)
        if isinstance(obj, dict) and obj.get("category") in CATEGORIES:
            return "classified_json", obj["category"]
        return "flagged_invalid_output", None
    except (json.JSONDecodeError, TypeError):
        found = [c for c in CATEGORIES if c in re.findall(r"[a-z]+", (content or "").lower())]
        return ("classified_keyword", found[0]) if len(found) == 1 else ("flagged_invalid_output", None)

def run_pipeline(tickets, system_prompt, token_budget=None, verbose=False):
    mock_llm.reset()
    stats, log, rows, sent_prompts, total = Counter(), [], [], [], 0
    for t in tickets.itertuples():
        text = t.text; used = 0; cat = None
        if not text.strip(): status = "skipped_empty"
        elif est_tokens(text) > MAX_INPUT_TOKENS: status = "flagged_long"
        elif token_budget is not None and total >= token_budget: status = "budget_exhausted"
        else:
            red = redact(text)
            msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": red}]
            for _, pat, _ in PATTERNS: assert not pat.search(msgs[1]["content"]), "PII reached the prompt"
            sent_prompts.append(msgs)
            try:
                r = call_with_retry(msgs, t.ticket_id, stats, log)
                used = r.usage.total_tokens; total += used
                status, cat = parse(r.choices[0].message.content)
            except TransientLLMError:
                status = "llm_error"
        rows.append({"ticket_id": t.ticket_id, "customer_id": t.customer_id, "status": status, "category": cat,
                     "tokens_used": used, "true_category": t.true_category})
    return pd.DataFrame(rows), stats, log, sent_prompts, total

def route(df, seg):
    seg = seg.set_index("customer_id")
    out = df.copy()
    out["segment_name"] = out.customer_id.map(seg.segment_name)
    prem = out.customer_id.map(seg.is_premium).fillna(False).astype(bool)
    human = out.status.isin(["skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted", "llm_error"])
    out["route"] = "support_queue"; out["priority"] = "NORMAL"
    out.loc[prem, ["route", "priority"]] = ["premium_desk", "HIGH"]
    fraud = (out.category == "fraud") & ~human
    out.loc[fraud, ["route", "priority"]] = ["security_team", "HIGH"]
    out.loc[human, ["route", "priority"]] = ["human_review", "NORMAL"]
    return out

if __name__ == "__main__":
    tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    print("=== C1 GUARDRAILS ===")
    print("empty:", int((tickets.text.str.strip() == "").sum()), "| long (>400 est tokens):",
          int(tickets.text.map(est_tokens).gt(MAX_INPUT_TOKENS).sum()), "| long estimate:", sorted(set(tickets.text.map(est_tokens)))[-1])
    print("\n=== C2 REDACTION ===")
    cnt = Counter(); red = tickets.text.map(lambda x: redact(x, cnt))
    print(dict(cnt), "total", sum(cnt.values()))
    print("patterns found in redacted texts:", sum(len(p.findall(x)) for x in red for _, p, _ in PATTERNS))
    print("\n=== C3 PROMPTS ===\nzero-shot est tokens:", est_tokens(ZERO_SHOT), "| few-shot est tokens:", est_tokens(FEW_SHOT))
    print("\n=== C4/C5/C6 FULL RUNS ===")
    res = {}
    for name, prompt in [("zero-shot", ZERO_SHOT), ("few-shot", FEW_SHOT)]:
        df, stats, log, sent, total = run_pipeline(tickets, prompt)
        res[name] = df
        waits = [float(l.split("wait=")[1][:-1]) for l in log]
        cl = df[df.status.str.startswith("classified")]
        acc = (cl.category == cl.true_category).mean()
        print(f"\n[{name}] transient errors={stats['transient_errors']} retries logged={len(log)} permanent errors={int((df.status=='llm_error').sum())}")
        print("  retry waits min %.4f max %.4f" % (min(waits), max(waits)), "| sample:", log[0])
        print("  status counts:", df.status.value_counts().to_dict())
        print(f"  classified={len(cl)} correct={int((cl.category==cl.true_category).sum())} accuracy={acc*100:.1f}% total_tokens={total} tokens/classified={total/len(cl):.1f} model calls={len(sent)}")
        if name == "zero-shot":
            print("  confusion (rows=true, cols=predicted):"); print(pd.crosstab(cl.true_category, cl.category).to_string())
        with open(f"outputs/retry_log_{name}.txt", "w") as f: f.write("\n".join(log) + "\n")
    print("\n=== C7 BUDGET + ROUTING ===")
    seg = pd.read_csv("outputs/customer_segments.csv")
    for budget in (6000,):
        df, stats, log, sent, total = run_pipeline(tickets, ZERO_SHOT, token_budget=budget)
        print(f"budget {budget}: budget_exhausted={int((df.status=='budget_exhausted').sum())}, tokens spent={total}, status counts {df.status.value_counts().to_dict()}")
        route(df, seg).drop(columns="true_category")[["ticket_id","customer_id","status","category","segment_name","priority","route","tokens_used"]].to_csv("outputs/triage_budget_demo.csv", index=False)
    df, stats, log, sent, total = run_pipeline(tickets, FEW_SHOT, token_budget=100000)
    final = route(df, seg)[["ticket_id","customer_id","status","category","segment_name","priority","route","tokens_used"]]
    final.to_csv("outputs/triage_results.csv", index=False)
    print("final rows", len(final), "| routes:", final.route.value_counts().to_dict(), "| priority:", final.priority.value_counts().to_dict())
    print("final status counts:", final.status.value_counts().to_dict())
    print("rule-validated (classified_json + classified_keyword):", int(final.status.str.startswith('classified').sum()), "| to human review:", int((final.route=="human_review").sum()))
