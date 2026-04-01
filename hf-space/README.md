---
title: Email Triage OpenEnv
emoji: 📧
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - email
  - triage
  - agent
  - reinforcement-learning
---

# Email Triage OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compliant-blue)](https://openenv.ai)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-green)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

A real-world OpenEnv environment where AI agents learn to manage professional
email inboxes. Agents classify priority, draft replies for urgent messages,
and archive junk — simulating a task every knowledge worker does daily.

---

## Environment description and motivation

Email triage is one of the most universal professional tasks. It requires:
- **Reading comprehension** — understanding urgency from natural language
- **Prioritization judgment** — server outage vs lunch invite
- **Professional communication** — contextually appropriate replies
- **Workflow efficiency** — clearing the inbox under a step budget

---

## Observation space

| Field | Type | Description |
|---|---|---|
| `inbox` | `List[Email]` | All emails in the episode |
| `processed` | `List[EmailState]` | Agent decisions so far |
| `current_email` | `Email` or null | Next unclassified email |
| `step_number` | int | Steps taken so far |
| `max_steps` | int | Step budget for this task |
| `task_id` | str | task1 / task2 / task3 |
| `task_description` | str | Full natural language objective |
| `available_actions` | List[str] | Valid action types |

**Email fields:** `id`, `sender`, `subject`, `body`, `timestamp`

---

## Action space

Send JSON to `POST /step`:

| `action_type` | Required fields | Description |
|---|---|---|
| `classify` | `email_id`, `priority` | Label email priority |
| `reply` | `email_id`, `reply_text` | Draft reply (min 10 words) |
| `archive` | `email_id` | Archive the email |
| `delete` | `email_id` | Delete the email |
| `read` | `email_id` | Neutral read action |
| `skip` | `email_id` | Neutral skip action |

**Priority values:** `urgent` | `high` | `normal` | `low` | `spam`

---

## Task descriptions with expected difficulty

| Task | Difficulty | Inbox | Steps | Scoring |
|---|---|---|---|---|
| task1 | Easy | 10 emails | 20 | 100% classification accuracy |
| task2 | Medium | 15 emails | 40 | 50% classify + 50% reply coverage |
| task3 | Hard | 20 emails | 60 | 40% classify + 35% reply + 25% archive |

---

## Setup and usage instructions

### Quick start (local)

```bash
git clone https://github.com/Neerav02/email-triage-openenv
cd email-triage-openenv
pip install flask requests openai PyYAML
python api/server.py
```

### Docker

```bash
docker build -t email-triage-openenv .
docker run -p 7860:7860 -e GROK_API_KEY=xai-... email-triage-openenv
```

### Test all endpoints

```bash
curl http://localhost:7860/health
curl -X POST http://localhost:7860/reset -H "Content-Type: application/json" -d '{"task_id":"task1"}'
curl http://localhost:7860/tasks
curl -X POST http://localhost:7860/grader
```

---

## API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/` | Info + endpoint list |
| GET | `/health` | Health ping |
| POST | `/reset` | Start episode |
| POST | `/step` | Execute action |
| GET | `/state` | Current episode snapshot |
| GET | `/tasks` | All tasks + action schema |
| POST | `/grader` | Score current episode |
| POST | `/baseline` | Run baseline (needs GROK_API_KEY) |

---

## Baseline scores (grok-3-mini / xAI Grok)

| Task | Difficulty | Score |
|---|---|---|
| task1 | Easy | 0.82 |
| task2 | Medium | 0.68 |
| task3 | Hard | 0.55 |

---

## Running tests

```bash
python tests/test_env.py
```

All 68 tests pass.

---

## License

MIT
