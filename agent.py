"""Part D tasks 5-9: tools, ReAct loop, safety limits, checkpoint/resume."""
import json, os, sqlite3, time, uuid
from datetime import datetime, timezone
import mock_llm
from rag import retrieve

DB, LEDGER, STATE_DIR = "data/instamart.db", "outputs/refund_ledger.jsonl", "state"

def get_order(order_id):
    con = sqlite3.connect(DB)
    try:
        row = con.execute("SELECT order_id, status, order_amount, delivery_minutes FROM orders WHERE order_id = ?", (order_id,)).fetchone()
    finally:
        con.close()
    if row is None: return {"error": "order not found"}
    return {"order_id": row[0], "status": row[1], "order_amount": float(row[2]),
            "delivery_minutes": None if row[3] is None or row[3] != row[3] else float(row[3])}

def search_policy(query):
    return {"results": retrieve(query)}

def calculate_refund(order_amount, delivery_minutes):
    """Late (>45 min) completed orders get 100% back; otherwise 0. (Status is checked by get_order/agent flow.)"""
    if delivery_minutes is None or delivery_minutes <= 45: return {"refund_inr": 0}
    return {"refund_inr": round(float(order_amount), 2)}

def _ledger(tool, order_id, refund_inr, run_id):
    os.makedirs(os.path.dirname(LEDGER), exist_ok=True)
    with open(LEDGER, "a") as f:
        f.write(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), "tool": tool, "run_id": run_id,
                            "order_id": order_id, "refund_inr": refund_inr}) + "\n")
    return {"refund_inr": refund_inr}

_CURRENT_RUN = {"id": None}
def issue_refund(order_id, refund_inr): return _ledger("issue_refund", order_id, refund_inr, _CURRENT_RUN["id"])
def escalate_to_human(order_id, refund_inr): return _ledger("escalate_to_human", order_id, refund_inr, _CURRENT_RUN["id"])

def _spec(name, desc, props): return {"name": name, "description": desc, "parameters": {"type": "object", "properties": props, "required": list(props)}}
TOOLS_SPEC = [
    _spec("get_order", "Look up an order's status, amount and delivery time in the orders database.", {"order_id": {"type": "string"}}),
    _spec("search_policy", "Search the policy documents and return the most relevant chunks.", {"query": {"type": "string"}}),
    _spec("calculate_refund", "Compute the refund in INR from the order amount and delivery minutes.", {"order_amount": {"type": "number"}, "delivery_minutes": {"type": "number"}}),
    _spec("issue_refund", "Issue an automatic refund (up to INR 1000) and record it in the ledger.", {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
    _spec("escalate_to_human", "Escalate a refund above INR 1000 to a human support lead.", {"order_id": {"type": "string"}, "refund_inr": {"type": "number"}}),
]
TOOL_MAP = {"get_order": get_order, "search_policy": search_policy, "calculate_refund": calculate_refund,
            "issue_refund": issue_refund, "escalate_to_human": escalate_to_human}

AGENT_SYSTEM = ("You are an AGENT that handles refund requests. Use the tools: get_order, search_policy, calculate_refund, "
                "issue_refund, escalate_to_human. Finish with one line 'STATUS: explanation' where STATUS is REFUND, ESCALATE, DENIED or NOT_FOUND.")

class SimulatedCrash(Exception): pass

def _save_checkpoint(state):
    os.makedirs(STATE_DIR, exist_ok=True)
    with open(os.path.join(STATE_DIR, f"{state['run_id']}.json"), "w") as f:
        json.dump(state, f, default=str)

def _finish(state, status, answer):
    refund = 0.0
    for s in state["steps"]:
        if s["tool"] in ("issue_refund", "escalate_to_human"): refund = s["observation"].get("refund_inr", 0)
    return {"run_id": state["run_id"], "scenario_id": state["scenario_id"], "status": status, "final_answer": answer,
            "steps": state["steps"], "llm_calls": state["llm_calls"], "tokens_used": state["tokens_used"], "refund_inr": refund}

def _fallback(state, status):
    done = [s["tool"] for s in state["steps"]]
    summary = (f"{status}: automatic handling stopped after {state['llm_calls']} model call(s) and {state['tokens_used']} tokens. "
               f"Tools completed so far: {', '.join(done) or 'none'}. Please hand this case to a human agent.")
    return _finish(state, status, summary)

def _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools):
    t0 = time.monotonic(); _CURRENT_RUN["id"] = state["run_id"]
    while True:
        if time.monotonic() - t0 >= time_limit_seconds: return _fallback(state, "FALLBACK_TIMEOUT")
        if state["llm_calls"] >= max_iterations: return _fallback(state, "FALLBACK_MAX_ITERATIONS")
        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1; state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > token_limit: return _fallback(state, "FALLBACK_TOKEN_LIMIT")   # tool NOT executed
        msg = resp.choices[0].message
        if not msg.tool_calls:
            return _finish(state, (msg.content or "").split(":")[0].strip(), msg.content)
        tc = msg.tool_calls[0]
        state["messages"].append({"role": "assistant", "content": None,
                                  "tool_calls": [{"id": tc.id, "name": tc.name, "arguments": tc.arguments}]})
        fn = TOOL_MAP.get(tc.name)
        try: obs = fn(**tc.arguments) if fn else {"error": f"unknown tool {tc.name}"}
        except Exception as e: obs = {"error": str(e)}
        state["messages"].append({"role": "tool", "tool_call_id": tc.id, "name": tc.name, "content": json.dumps(obs)})
        state["steps"].append({"tool": tc.name, "arguments": tc.arguments, "observation": obs})
        state["tools_executed"].append(tc.name); state["step"] += 1
        _save_checkpoint(state)
        if crash_after_tools is not None and state["step"] == crash_after_tools:
            raise SimulatedCrash(f"simulated crash after tool #{state['step']} ({tc.name})")

def run_agent(user_message, scenario_id, max_iterations=6, token_limit=6000, time_limit_seconds=10, crash_after_tools=None):
    state = {"run_id": f"{scenario_id}-{uuid.uuid4().hex[:8]}", "scenario_id": scenario_id,
             "messages": [{"role": "system", "content": AGENT_SYSTEM}, {"role": "user", "content": user_message}],
             "tokens_used": 0, "llm_calls": 0, "step": 0, "tools_executed": [], "steps": []}
    return _loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tools)

def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(os.path.join(STATE_DIR, f"{run_id}.json")) as f: state = json.load(f)
    return _loop(state, max_iterations, token_limit, time_limit_seconds, None)
