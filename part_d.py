"""Part D runner."""
import json, os, shutil
import pandas as pd
from rag import chunk_documents, Retriever, retrieve, evaluate, answer
import agent
from agent import run_agent, resume_agent, SimulatedCrash

print("=== D1 CHUNKS ===")
ch = chunk_documents(); print("total", len(ch), pd.Series([c["source"] for c in ch]).value_counts().sort_index().to_dict())
print("\n=== D2 RETRIEVAL ===")
print([ (h["source"], round(h["score"],3)) for h in retrieve("refund policy late delivery")])
print("\n=== D4 EVAL (overlap 10) ===")
r, fails = evaluate(); print(r, "failures:", fails)
print("\n=== D4 EVAL (overlap 0) ===")
r0, fails0 = evaluate(Retriever(40, 0)); print(r0, "chunks:", len(Retriever(40,0).chunks))
for q, why, txt in fails0: print("  FAIL:", q, "|", why, "| answer:", txt)

# fresh outputs so reruns are idempotent
for p in ("outputs/agent_runs.jsonl", "outputs/refund_ledger.jsonl", "outputs/agent_demos.jsonl"):
    if os.path.exists(p): os.remove(p)
shutil.rmtree("state", ignore_errors=True); os.makedirs("state")

print("\n=== D6 SEVEN SCENARIOS ===")
sc = pd.read_csv("data/raw/agent_scenarios.csv"); rows = []
with open("outputs/agent_runs.jsonl", "a") as f:
    for s in sc.itertuples():
        res = run_agent(s.message, s.scenario_id)
        f.write(json.dumps({k: res[k] for k in ("scenario_id", "status", "final_answer", "llm_calls", "tokens_used", "steps")}) + "\n")
        exp = s.expected_outcome
        ok = res["status"] == ("FALLBACK_MAX_ITERATIONS" if exp == "FALLBACK" else exp) and abs(res["refund_inr"] - s.expected_refund_inr) <= 0.01
        rows.append((s.scenario_id, res["status"], f"{res['refund_inr']:.2f}", res["llm_calls"], res["tokens_used"], ok))
print(pd.DataFrame(rows, columns=["scenario", "status", "refund_inr", "model_calls", "tokens", "matches_expected"]).to_string(index=False))
assert all(r[-1] for r in rows)

print("\n=== D7 STOP SAFELY ===")
demos = open("outputs/agent_demos.jsonl", "a")
s1 = sc.iloc[0]
a = run_agent(s1.message, "S1-tokenlimit", token_limit=200)
print("S1 token_limit=200 ->", a["status"], "| tokens", a["tokens_used"], "| tools run:", [s["tool"] for s in a["steps"]]); print("   handoff:", a["final_answer"])
b = run_agent(s1.message, "S1-timeout", time_limit_seconds=0)
print("S1 time_limit_seconds=0 ->", b["status"], "| model calls", b["llm_calls"]); print("   handoff:", b["final_answer"])

print("\n=== D8 CRASH AND RESUME ===")
run_id = None
try:
    run_agent(s1.message, "S1-resume", crash_after_tools=2)
except SimulatedCrash as e:
    print("crash:", e)
run_id = [f[:-5] for f in os.listdir("state") if f.startswith("S1-resume")][0]
ck = json.load(open(f"state/{run_id}.json")); print("checkpoint tools_executed:", ck["tools_executed"], "step", ck["step"], "tokens", ck["tokens_used"])
c = resume_agent(run_id)
print("resumed ->", c["status"], c["refund_inr"], "| tools across both parts:", c["steps"] and [s["tool"] for s in c["steps"]])
assert [s["tool"] for s in c["steps"]] == ["get_order", "search_policy", "calculate_refund", "issue_refund"]
assert c["status"] == "REFUND" and abs(c["refund_inr"] - 128.25) < 0.01
for name, res in [("tokenlimit", a), ("timeout", b), ("resume", c)]:
    demos.write(json.dumps({"demo": name, **{k: res[k] for k in ("run_id", "status", "final_answer", "llm_calls", "tokens_used", "steps")}}) + "\n")
demos.close()
print("\nledger lines:", sum(1 for _ in open("outputs/refund_ledger.jsonl")), "| state files:", len(os.listdir("state")))
