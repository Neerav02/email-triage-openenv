import os
import sys
import json
import time
import requests

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.x.ai/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "grok-3-mini")
HF_TOKEN = os.getenv("HF_TOKEN", "")
ENV_API_BASE = os.getenv("ENV_API_BASE", "http://localhost:7860")
BENCHMARK = "email-triage-openenv"

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

def log_start(task: str, env: str, model: str):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: str = None):
    err_str = f" error={error}" if error else ""
    print(f"[STEP] step={step} action={repr(action)} reward={reward:+.2f} done={done}{err_str}", flush=True)

def log_end(success: bool, steps: int, score: float, rewards: list):
    print(f"[END] success={success} steps={steps} score={score:.2f} rewards={rewards}", flush=True)

def get_client():
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: openai package not installed. Run: pip install openai", file=sys.stderr)
        sys.exit(0)
    
    key = HF_TOKEN or "dummy_key"
    if not HF_TOKEN:
        print("WARNING: HF_TOKEN not set. Using dummy key to prevent crash.", file=sys.stderr)
        
    try:
        return OpenAI(api_key=key, base_url=API_BASE_URL)
    except Exception as e:
        print(f"ERROR: Failed to init client: {e}", file=sys.stderr)
        return None

def run_task(client, task_id: str, model: str) -> float:
    try:
        r = requests.post(f"{ENV_API_BASE}/reset", json={"task_id": task_id}, timeout=15)
        r.raise_for_status()
        obs = r.json()
    except Exception as e:
        print(f"  ERROR connecting to {ENV_API_BASE}: {e}", file=sys.stderr)
        log_start(task=task_id, env=BENCHMARK, model=model)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return 0.0

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
    rewards = []
    
    log_start(task=task_id, env=BENCHMARK, model=model)

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

        error_msg = None
        action_text = ""
        action_json = None
        
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
        except json.JSONDecodeError as decode_err:
            error_msg = f"JSONDecodeError: {decode_err}"
            errors += 1
            log_step(step=step, action=action_text, reward=0.0, done=False, error=error_msg)
            if errors >= 5: break
            messages.append({"role": "user", "content": "Invalid JSON. Respond with ONLY a JSON object."})
            continue
        except Exception as e:
            error_msg = f"APIError: {e}"
            errors += 1
            log_step(step=step, action="", reward=0.0, done=False, error=error_msg)
            if errors >= 5: break
            time.sleep(2)
            continue

        try:
            r = requests.post(f"{ENV_API_BASE}/step", json=action_json, timeout=15)
            result = r.json()
            if "error" in result.get("info", {}):
                error_msg = result['info']['error']
                messages.append({"role": "user", "content": f"Error: {error_msg}. Try again."})
                errors += 1
                log_step(step=step, action=action_text, reward=0.0, done=False, error=error_msg)
                continue
        except Exception as e:
            error_msg = f"Env connection error: {e}"
            log_step(step=step, action=action_text, reward=0.0, done=False, error=error_msg)
            break

        obs    = result["observation"]
        done   = result["done"]
        reward_dict = result.get("reward", {})
        total_reward = float(reward_dict.get("total", 0.0) if type(reward_dict) is dict else reward_dict)

        rewards.append(total_reward)
        log_step(step=step, action=action_text, reward=total_reward, done=done, error=None)

        if done:
            final_score = float(reward_dict.get("components", {}).get("final_grader_score", 0.0) if type(reward_dict) is dict else total_reward)
            break

        messages.append({"role": "user", "content": f"reward={total_reward:.3f}. Continue."})

    if not done:
        try:
            r = requests.post(f"{ENV_API_BASE}/grader", timeout=10)
            final_score = float(r.json().get("score", 0.0))
        except Exception:
            pass

    success = final_score >= 0.8
    log_end(success=success, steps=step, score=final_score, rewards=rewards)
    return round(final_score, 4)


def main():
    try:
        for _ in range(3):
            try:
                r = requests.get(f"{ENV_API_BASE}/health", timeout=10)
                r.raise_for_status()
                break
            except Exception:
                time.sleep(2)
        else:
            r = requests.get(f"{ENV_API_BASE}/health", timeout=10)
            r.raise_for_status()
    except Exception as e:
        print(f"ERROR: Cannot reach {ENV_API_BASE}: {e}", file=sys.stderr)
        print(json.dumps({"task1": 0.0, "task2": 0.0, "task3": 0.0}))
        sys.exit(0)

    client = get_client()
    if client is None:
        print(json.dumps({"task1": 0.0, "task2": 0.0, "task3": 0.0}))
        sys.exit(0)

    try:
        tasks = ["task1", "task2", "task3"]
        scores = {}
        for task_id in tasks:
            score = run_task(client, task_id, MODEL_NAME)
            scores[task_id] = score

        print(json.dumps(scores))
        return scores
    except Exception as e:
        print(f"ERROR: Unhandled loop exception: {e}", file=sys.stderr)
        print(json.dumps({"task1": 0.0, "task2": 0.0, "task3": 0.0}))
        sys.exit(0)

if __name__ == "__main__":
    main()
