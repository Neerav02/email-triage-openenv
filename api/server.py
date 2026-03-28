"""
Email Triage OpenEnv - Flask API Server
Implements all required OpenEnv endpoints:
  GET  /          - info
  GET  /health    - ping
  POST /reset     - start episode
  POST /step      - execute action
  GET  /state     - current state
  GET  /tasks     - task list + action schema
  POST /grader    - score current episode
  POST /baseline  - run baseline script (uses Grok API)
"""
import json
import os
import subprocess
import sys

from flask import Flask, request, jsonify

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
            "GET  /",
            "GET  /health",
            "POST /reset   body: {task_id: task1|task2|task3}",
            "POST /step    body: {action_type, email_id, priority?, reply_text?}",
            "GET  /state",
            "GET  /tasks",
            "POST /grader",
            "POST /baseline",
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


@app.post("/step")
def step():
    data = request.get_json(force=True, silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400
    action = Action.from_dict(data)
    if not action.action_type:
        return jsonify({"error": "action_type is required"}), 400
    if not action.email_id:
        return jsonify({"error": "email_id is required"}), 400
    try:
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
        return jsonify({"error": f"Internal error: {str(e)}"}), 500


@app.get("/state")
def state():
    return jsonify(env.state())


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
        return jsonify({"error": "No active episode. Call /reset first."}), 400
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
    """
    Trigger baseline inference using Grok API.
    Requires GROK_API_KEY environment variable.
    """
    api_key = os.environ.get("GROK_API_KEY", "")
    if not api_key or api_key.startswith("xai-your"):
        return jsonify({
            "message": "GROK_API_KEY not set. Returning documented baseline scores.",
            "baseline_scores": {"task1": 0.82, "task2": 0.68, "task3": 0.55},
            "provider": "xAI Grok",
            "note": "Set GROK_API_KEY secret and call again to run live baseline.",
        })

    try:
        result = subprocess.run(
            [sys.executable, "baseline_inference.py", "--mode", "api"],
            capture_output=True, text=True, timeout=300,
            env={**os.environ, "ENV_API_BASE": "http://localhost:7860"},
        )
        if result.returncode != 0:
            return jsonify({"error": "Baseline script failed", "detail": result.stderr[:500]}), 500
        lines = [l.strip() for l in result.stdout.strip().split("\n") if l.strip()]
        scores = json.loads(lines[-1])
        return jsonify({"baseline_scores": scores, "provider": "xAI Grok"})
    except subprocess.TimeoutExpired:
        return jsonify({"error": "Baseline timed out (300s)"}), 504
    except json.JSONDecodeError:
        return jsonify({"error": "Could not parse baseline output"}), 500


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
    print(f"  Email Triage OpenEnv  |  http://localhost:{port}")
    print(f"{'='*55}")
    print(f"  GET  http://localhost:{port}/health")
    print(f"  GET  http://localhost:{port}/tasks")
    print(f"  POST http://localhost:{port}/reset")
    print(f"  POST http://localhost:{port}/step")
    print(f"  GET  http://localhost:{port}/state")
    print(f"  POST http://localhost:{port}/grader")
    print(f"  POST http://localhost:{port}/baseline  (needs GROK_API_KEY)")
    print(f"{'='*55}\n")
    app.run(host="0.0.0.0", port=port, debug=False)
