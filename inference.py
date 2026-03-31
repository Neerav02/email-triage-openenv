"""
Baseline inference script for Email Triage OpenEnv.
Uses Grok API (xAI) via the OpenAI-compatible client.

Grok is fully compatible with the OpenAI Python SDK —
just needs a different base_url and model name.

Usage:
    python baseline_inference.py                    # human-readable output
    python baseline_inference.py --mode api         # outputs JSON on last line
    python baseline_inference.py --task task1       # single task only
    python baseline_inference.py --model grok-3-mini

Environment variables:
    GROK_API_KEY   required - your xAI Grok API key (starts with xai-)
    ENV_API_BASE   optional - env server URL (default: http://localhost:7860)

Get your key at: https://console.x.ai/
"""
import os
import sys
import json
import argparse
import time
import requests

GROK_API_KEY  = os.getenv("GROK_API_KEY", "")
GROK_BASE_URL = "https://api.x.ai/v1"
DEFAULT_MODEL = "grok-3-mini"
ENV_API_BASE  = os.getenv("ENV_API_BASE", "http://localhost:7860")

SYSTEM_PROMPT = """You are an expert email assistant helping manage a professional inbox.

Take ONE action at a time. Respond with ONLY a valid JSON object.

ACTIONS:
{"action_type": "classify", "email_id": "<id>", "priority": "<urgent|high|normal|low|spam>"}
{"action_type": "reply",    "email_id": "<id>", "reply_text": "<professional reply, minimum 15 words>"}
{"action_type": "archive",  "email_id": "<id>"}
{"action_type": "delete",   "email_id": "<id>"}
{"action_type": "skip",     "email_id": "<id>"}

PRIORITY RULES:
- urgent: immediate action required, major business impact, time-critical
- high:   important, needs response today
- normal: standard communication, respond within days
- low:    informational only, no action required
- spam:   unsolicited, scam, phishing, marketing

STRATEGY:
1. Classify ALL emails first
2. Draft replies for urgent and high priority emails (15+ words)
3. Archive low priority, delete spam
4. One action per turn. Respond with ONLY valid JSON.
"""


def get_client():
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(1)
    if not GROK_API_KEY:
        print("ERROR: GROK_API_KEY not set. Get key at https://console.x.ai/", file=sys.stderr)
        sys.exit(1)
    return OpenAI(api_key=GROK_API_KEY, base_url=GROK_BASE_URL)


def run_task(client, task_id: str, model: str, verbose: bool = True) -> float:
    try:
        r = requests.post(f"{ENV_API_BASE}/reset", json={"task_id": task_id}, timeout=15)
        r.raise_for_status()
        obs = r.json()
    except Exception as e:
        print(f"  ERROR connecting to {ENV_API_BASE}: {e}", file=sys.stderr)
        return 0.0

    if verbose:
        print(f"  Inbox: {len(obs['inbox'])} emails | Max steps: {obs['max_steps']}")

    inbox_text = "\n\n".join(
        f"ID: {e['id']}\nFrom: {e['sender']}\nSubject: {e['subject']}\nBody: {e['body'][:250]}"
        for e in obs["inbox"]
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": (
            f"TASK: {obs['task_description']}\n\n"
            f"INBOX ({len(obs['inbox'])} emails):\n\n{inbox_text}\n\n"
            "Start classifying. First action JSON:"
        )},
    ]

    done = False
    step = 0
    final_score = 0.0
    errors = 0

    while not done and step < obs["max_steps"] + 5:
        step += 1

        if step > 1:
            current = obs.get("current_email")
            unclassified = sum(1 for e in obs.get("processed", []) if e.get("priority") is None)
            hint = f"Step {step}."
            if current:
                hint += f" Focus: ID={current['id']} Subject='{current['subject'][:60]}'"
            hint += f" Unclassified: {unclassified}. Next JSON action:"
            messages.append({"role": "user", "content": hint})

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.1,
                max_tokens=300,
                response_format={"type": "json_object"},
            )
            action_text = completion.choices[0].message.content.strip()
            action_json = json.loads(action_text)
            messages.append({"role": "assistant", "content": action_text})
            errors = 0
        except json.JSONDecodeError:
            errors += 1
            if errors >= 5: break
            messages.append({"role": "user", "content": "Invalid JSON. Respond with ONLY a JSON object."})
            continue
        except Exception as e:
            errors += 1
            if verbose:
                print(f"    Grok error at step {step}: {e}", file=sys.stderr)
            if errors >= 5: break
            time.sleep(2)
            continue

        try:
            r = requests.post(f"{ENV_API_BASE}/step", json=action_json, timeout=15)
            result = r.json()
            if "error" in result.get("info", {}):
                messages.append({"role": "user", "content": f"Error: {result['info']['error']}. Try again."})
                errors += 1
                continue
        except Exception as e:
            print(f"    Step error: {e}", file=sys.stderr)
            break

        obs    = result["observation"]
        done   = result["done"]
        reward = result["reward"]

        if verbose and step % 5 == 0:
            classified = sum(1 for e in obs["processed"] if e.get("priority"))
            print(f"    step={step} classified={classified}/{len(obs['inbox'])} reward={reward['total']:.3f}")

        if done:
            final_score = reward.get("components", {}).get("final_grader_score", 0.0)
            if verbose:
                print(f"  Done at step {step}. Score: {final_score:.4f}")
            break

        messages.append({"role": "user", "content": f"reward={reward['total']:.3f}. Continue."})

    if not done:
        try:
            r = requests.post(f"{ENV_API_BASE}/grader", timeout=10)
            final_score = r.json().get("score", 0.0)
            if verbose:
                print(f"  Max steps reached. Grader score: {final_score:.4f}")
        except Exception:
            pass

    return round(final_score, 4)


def main():
    parser = argparse.ArgumentParser(description="Email Triage OpenEnv Baseline - Grok/xAI")
    parser.add_argument("--mode",  default="print", choices=["print", "api"])
    parser.add_argument("--task",  default="all", choices=["all", "task1", "task2", "task3"])
    parser.add_argument("--model", default=DEFAULT_MODEL)
    args = parser.parse_args()
    verbose = args.mode == "print"

    client = get_client()

    try:
        r = requests.get(f"{ENV_API_BASE}/health", timeout=10)
        r.raise_for_status()
        if verbose:
            print(f"Connected to {ENV_API_BASE}")
            print(f"Model: {args.model} | Provider: xAI Grok ({GROK_BASE_URL})")
    except Exception as e:
        print(f"ERROR: Cannot reach {ENV_API_BASE}: {e}", file=sys.stderr)
        sys.exit(1)

    tasks = ["task1", "task2", "task3"] if args.task == "all" else [args.task]
    difficulties = {"task1": "easy", "task2": "medium", "task3": "hard"}
    scores = {}

    for task_id in tasks:
        if verbose:
            print(f"\nRunning {task_id} ({difficulties[task_id]})...")
        score = run_task(client, task_id, args.model, verbose)
        scores[task_id] = score
        if verbose:
            bar = "#" * int(score * 30)
            print(f"  {task_id}: {score:.4f}  [{bar:<30}]")

    if verbose:
        print(f"\n{'='*50}")
        print("FINAL BASELINE SCORES (Grok/xAI):")
        for tid, s in scores.items():
            print(f"  {tid} ({difficulties[tid]:6}): {s:.4f}")
        print(f"{'='*50}")

    print(json.dumps(scores))
    return scores


if __name__ == "__main__":
    main()
