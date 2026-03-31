"""
Email Triage OpenEnv - Flask API Server
All required OpenEnv endpoints implemented.
"""
import json
import os
import subprocess
import sys

from flask import Flask, request, jsonify

# Ensure project root is on path regardless of working directory
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from env.environment import EmailTriageEnv
from env.models import Action, ACTION_SCHEMA
from env.tasks import TASKS

app = Flask(__name__)
env = EmailTriageEnv()


@app.get("/")
def root():
    return jsonify({
        "name": "email-triage-openenv",
        "version": "1.0.0",
        "status": "ready",
        "tag": "openenv",
        "description": "Real-world email triage environment for AI agent training and evaluation.",
        "tasks": list(TASKS.keys()),
        "endpoints": [
            "GET  /", "GET  /health",
            "POST /reset", "POST /step",
            "GET  /state", "GET  /tasks",
            "POST /grader", "POST /baseline",
        ],
    })


@app.get("/health")
def health():
    return jsonify({"status": "ok", "environment": "email-triage-openenv", "tag": "openenv"})


@app.post("/reset")
def reset():
    data = request.get_json(force=True, silent=True) or {}
    task_id = data.get("task_id", "task1")
    try:
        obs = env.reset(task_id=task_id)
        return jsonify(obs.to_dict())
    except ValueError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Reset failed: {str(e)}"}), 500


@app.post("/step")
def step():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    if not data.get("action_type"):
        return jsonify({"error": "action_type is required"}), 400
    if not data.get("email_id"):
        return jsonify({"error": "email_id is required"}), 400
    try:
        action = Action.from_dict(data)
        obs, reward, done, info = env.step(action)
        return jsonify({
            "observation": obs.to_dict(),
            "reward": reward.to_dict(),
            "done": done,
            "info": info,
        })
    except RuntimeError as e:
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        return jsonify({"error": f"Step failed: {str(e)}"}), 500


@app.get("/state")
def state():
    try:
        return jsonify(env.state())
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/tasks")
def list_tasks():
    return jsonify({
        task_id: {
            "id": task.id,
            "name": task.name,
            "description": task.description,
            "difficulty": task.difficulty,
            "max_steps": task.max_steps,
            "inbox_size": task.inbox_size,
            "action_schema": ACTION_SCHEMA,
            "scoring": _scoring_info(task_id),
        }
        for task_id, task in TASKS.items()
    })


@app.post("/grader")
def grader():
    if env.task_id is None:
        # Auto-reset so grader always works even without prior reset
        env.reset("task1")
    score = env.grade()
    s = env.state()
    return jsonify({
        "score": score,
        "task_id": env.task_id,
        "done": env.done,
        "step": env.step_count,
        "emails_classified": s["emails_classified"],
        "emails_replied": s["emails_replied"],
        "emails_archived": s["emails_archived"],
    })


@app.post("/baseline")
def baseline():
    api_key = os.environ.get("GROK_API_KEY", "")
    if not api_key or len(api_key) < 10:
        return jsonify({
            "message": "GROK_API_KEY not set. Returning documented baseline scores.",
            "baseline_scores": {"task1": 0.82, "task2": 0.68, "task3": 0.55},
            "provider": "xAI Grok",
            "note": "Set GROK_API_KEY secret in HF Space settings to run live baseline.",
        })
    try:
        script = os.path.join(ROOT, "baseline_inference.py")
        result = subprocess.run(
            [sys.executable, script, "--mode", "api"],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "ENV_API_BASE": "http://localhost:7860"},
        )
        if result.returncode != 0:
            return jsonify({"error": "Baseline failed", "detail": result.stderr[:500]}), 500
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        scores = json.loads(lines[-1])
        return jsonify({"baseline_scores": scores, "provider": "xAI Grok"})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Baseline timed out"}), 504
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def _scoring_info(task_id):
    info = {
        "task1": {"method": "% emails correctly classified",
                  "weights": {"classification": "100%"}},
        "task2": {"method": "50% classify + 50% reply coverage",
                  "weights": {"classification": "50%", "reply_coverage_urgent_high": "50%"}},
        "task3": {"method": "40% classify + 35% reply + 25% archive",
                  "weights": {"classification": "40%", "reply_quality": "35%", "archive_spam_low": "25%"}},
    }
    return info.get(task_id, {})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print(f"\n{'='*55}")
    print(f"  Email Triage OpenEnv  |  http://0.0.0.0:{port}")
    print(f"{'='*55}")
    for ep in ["GET /health", "GET /tasks", "POST /reset", "POST /step", "GET /state", "POST /grader", "POST /baseline"]:
        print(f"  {ep}")
    print(f"{'='*55}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
