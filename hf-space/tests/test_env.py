"""
Full test suite for Email Triage OpenEnv.
Uses stdlib unittest only (no pytest required).
Run with: python -m pytest tests/ -v
      or: python tests/test_env.py
"""
import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from env.environment import EmailTriageEnv
from env.models import Action, EmailState
from env.tasks import TASKS, grade_task1, grade_task2, grade_task3
from env.data_generator import generate_inbox, generate_email


# ======================================================================
# reset() tests
# ======================================================================

class TestReset(unittest.TestCase):

    def test_task1_inbox_size(self):
        env = EmailTriageEnv()
        obs = env.reset("task1")
        self.assertEqual(len(obs.inbox), 10)

    def test_task2_inbox_size(self):
        env = EmailTriageEnv()
        obs = env.reset("task2")
        self.assertEqual(len(obs.inbox), 15)

    def test_task3_inbox_size(self):
        env = EmailTriageEnv()
        obs = env.reset("task3")
        self.assertEqual(len(obs.inbox), 20)

    def test_step_starts_at_zero(self):
        env = EmailTriageEnv()
        obs = env.reset("task1")
        self.assertEqual(obs.step_number, 0)

    def test_max_steps_task1(self):
        env = EmailTriageEnv()
        obs = env.reset("task1")
        self.assertEqual(obs.max_steps, 20)

    def test_max_steps_task2(self):
        env = EmailTriageEnv()
        obs = env.reset("task2")
        self.assertEqual(obs.max_steps, 40)

    def test_max_steps_task3(self):
        env = EmailTriageEnv()
        obs = env.reset("task3")
        self.assertEqual(obs.max_steps, 60)

    def test_invalid_task_raises(self):
        env = EmailTriageEnv()
        with self.assertRaises(ValueError):
            env.reset("task99")

    def test_all_processed_start_unclassified(self):
        env = EmailTriageEnv()
        obs = env.reset("task1")
        for state in obs.processed:
            self.assertIsNone(state.priority)
            self.assertIsNone(state.reply_draft)
            self.assertFalse(state.archived)
            self.assertFalse(state.deleted)

    def test_reset_clears_prior_episode(self):
        env = EmailTriageEnv()
        obs1 = env.reset("task1")
        email_id = obs1.inbox[0].id
        env.step(Action("classify", email_id, priority="urgent"))
        obs2 = env.reset("task1")
        self.assertEqual(obs2.step_number, 0)
        self.assertTrue(all(e.priority is None for e in obs2.processed))

    def test_available_actions_present(self):
        env = EmailTriageEnv()
        obs = env.reset("task1")
        expected = {"classify", "reply", "archive", "delete", "read", "skip"}
        self.assertEqual(set(obs.available_actions), expected)

    def test_current_email_set(self):
        env = EmailTriageEnv()
        obs = env.reset("task1")
        self.assertIsNotNone(obs.current_email)

    def test_task_description_nonempty(self):
        env = EmailTriageEnv()
        obs = env.reset("task1")
        self.assertGreater(len(obs.task_description), 20)


# ======================================================================
# step() tests
# ======================================================================

class TestStep(unittest.TestCase):

    def setUp(self):
        self.env = EmailTriageEnv()
        self.obs = self.env.reset("task1")
        self.eid = self.obs.inbox[0].id

    def test_step_returns_four_tuple(self):
        action = Action("classify", self.eid, priority="urgent")
        result = self.env.step(action)
        self.assertEqual(len(result), 4)

    def test_step_increments_step_number(self):
        action = Action("classify", self.eid, priority="spam")
        obs, _, _, _ = self.env.step(action)
        self.assertEqual(obs.step_number, 1)

    def test_reward_total_in_range(self):
        action = Action("classify", self.eid, priority="high")
        _, reward, _, _ = self.env.step(action)
        self.assertGreaterEqual(reward.total, 0.0)
        self.assertLessEqual(reward.total, 1.0)

    def test_classify_sets_priority(self):
        action = Action("classify", self.eid, priority="urgent")
        obs, _, _, _ = self.env.step(action)
        state = next(e for e in obs.processed if e.id == self.eid)
        self.assertEqual(state.priority, "urgent")

    def test_reply_sets_draft(self):
        reply = "Thank you for your email I will look into this right away and respond shortly."
        action = Action("reply", self.eid, reply_text=reply)
        obs, _, _, _ = self.env.step(action)
        state = next(e for e in obs.processed if e.id == self.eid)
        self.assertEqual(state.reply_draft, reply)

    def test_archive_sets_flag(self):
        action = Action("archive", self.eid)
        obs, _, _, _ = self.env.step(action)
        state = next(e for e in obs.processed if e.id == self.eid)
        self.assertTrue(state.archived)

    def test_delete_sets_flag(self):
        action = Action("delete", self.eid)
        obs, _, _, _ = self.env.step(action)
        state = next(e for e in obs.processed if e.id == self.eid)
        self.assertTrue(state.deleted)

    def test_invalid_email_id_returns_error(self):
        action = Action("classify", "bad_id_xyz", priority="spam")
        _, _, done, info = self.env.step(action)
        self.assertIn("error", info)
        self.assertFalse(done)

    def test_step_after_done_raises(self):
        env = EmailTriageEnv()
        env.reset("task1")
        env.done = True
        with self.assertRaises(RuntimeError):
            env.step(Action("skip", "any"))

    def test_task1_done_when_all_classified(self):
        env = EmailTriageEnv()
        obs = env.reset("task1")
        done = False
        for email in obs.inbox:
            _, _, done, _ = env.step(Action("classify", email.id, priority="normal"))
        self.assertTrue(done)

    def test_done_at_max_steps(self):
        env = EmailTriageEnv()
        env.reset("task1")
        eid = env.inbox[0].id
        done = False
        for _ in range(25):
            _, _, done, _ = env.step(Action("skip", eid))
            if done:
                break
        self.assertTrue(done)

    def test_reward_components_is_dict(self):
        action = Action("classify", self.eid, priority="urgent")
        _, reward, _, _ = self.env.step(action)
        self.assertIsInstance(reward.components, dict)
        self.assertGreater(len(reward.components), 0)

    def test_skip_action_neutral(self):
        action = Action("skip", self.eid)
        _, reward, _, _ = self.env.step(action)
        self.assertGreaterEqual(reward.total, 0.0)


# ======================================================================
# state() tests
# ======================================================================

class TestState(unittest.TestCase):

    def test_state_returns_dict(self):
        env = EmailTriageEnv()
        env.reset("task1")
        self.assertIsInstance(env.state(), dict)

    def test_state_required_fields(self):
        env = EmailTriageEnv()
        env.reset("task1")
        s = env.state()
        for f in ["task_id", "step", "done", "inbox_size",
                  "emails_classified", "emails_replied", "emails_archived",
                  "cumulative_reward", "processed", "action_history"]:
            self.assertIn(f, s, f"Missing field: {f}")

    def test_state_step_increments(self):
        env = EmailTriageEnv()
        obs = env.reset("task1")
        self.assertEqual(env.state()["step"], 0)
        env.step(Action("skip", obs.inbox[0].id))
        self.assertEqual(env.state()["step"], 1)

    def test_state_emails_classified_count(self):
        env = EmailTriageEnv()
        obs = env.reset("task1")
        self.assertEqual(env.state()["emails_classified"], 0)
        env.step(Action("classify", obs.inbox[0].id, priority="spam"))
        self.assertEqual(env.state()["emails_classified"], 1)

    def test_state_task_id_matches(self):
        env = EmailTriageEnv()
        env.reset("task2")
        self.assertEqual(env.state()["task_id"], "task2")


# ======================================================================
# Grader tests
# ======================================================================

class TestGraders(unittest.TestCase):

    def test_task1_all_correct(self):
        gt = {"e1": "urgent", "e2": "spam", "e3": "normal"}
        states = [
            EmailState("e1", priority="urgent"),
            EmailState("e2", priority="spam"),
            EmailState("e3", priority="normal"),
        ]
        self.assertAlmostEqual(grade_task1(states, gt), 1.0)

    def test_task1_all_wrong(self):
        gt = {"e1": "urgent", "e2": "spam"}
        states = [
            EmailState("e1", priority="low"),
            EmailState("e2", priority="high"),
        ]
        self.assertAlmostEqual(grade_task1(states, gt), 0.0)

    def test_task1_half_correct(self):
        gt = {"e1": "urgent", "e2": "spam"}
        states = [
            EmailState("e1", priority="urgent"),
            EmailState("e2", priority="high"),
        ]
        self.assertAlmostEqual(grade_task1(states, gt), 0.5)

    def test_task1_none_classified(self):
        gt = {"e1": "urgent"}
        states = [EmailState("e1")]
        self.assertAlmostEqual(grade_task1(states, gt), 0.0)

    def test_task1_empty(self):
        self.assertAlmostEqual(grade_task1([], {}), 0.0)

    def test_task2_perfect_score(self):
        gt = {"e1": "urgent", "e2": "high", "e3": "normal"}
        reply = "I will handle this immediately and get back to you with a full update today."
        states = [
            EmailState("e1", priority="urgent", reply_draft=reply),
            EmailState("e2", priority="high", reply_draft=reply),
            EmailState("e3", priority="normal"),
        ]
        self.assertAlmostEqual(grade_task2(states, gt), 1.0)

    def test_task2_classify_only(self):
        gt = {"e1": "urgent", "e2": "high"}
        states = [
            EmailState("e1", priority="urgent"),
            EmailState("e2", priority="high"),
        ]
        # classify=1.0, reply=0.0 => 0.5*1 + 0.5*0 = 0.5
        self.assertAlmostEqual(grade_task2(states, gt), 0.5)

    def test_task2_score_range(self):
        env = EmailTriageEnv()
        env.reset("task2")
        score = env.grade()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_task3_perfect_score(self):
        gt = {"e1": "urgent", "e2": "high", "e3": "spam", "e4": "low"}
        reply = "I am treating this as top priority and will escalate internally to resolve within the hour."
        states = [
            EmailState("e1", priority="urgent", reply_draft=reply),
            EmailState("e2", priority="high", reply_draft=reply),
            EmailState("e3", priority="spam", deleted=True),
            EmailState("e4", priority="low", archived=True),
        ]
        self.assertAlmostEqual(grade_task3(states, gt), 1.0)

    def test_task3_weights(self):
        gt = {"e1": "urgent", "e2": "spam"}
        states = [
            EmailState("e1", priority="urgent"),   # right label, no reply
            EmailState("e2", priority="spam"),     # right label, not archived
        ]
        # classify=1.0, reply=0.0, archive=0.0
        expected = 0.40 * 1.0 + 0.35 * 0.0 + 0.25 * 0.0
        self.assertAlmostEqual(grade_task3(states, gt), expected, places=3)

    def test_task3_score_range(self):
        env = EmailTriageEnv()
        env.reset("task3")
        score = env.grade()
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_graders_deterministic(self):
        gt = {"e1": "urgent", "e2": "normal"}
        states = [EmailState("e1", priority="urgent"), EmailState("e2")]
        s1 = grade_task1(states, gt)
        s2 = grade_task1(states, gt)
        self.assertEqual(s1, s2)


# ======================================================================
# Data generator tests
# ======================================================================

class TestDataGenerator(unittest.TestCase):

    def test_inbox_exact_size_10(self):
        self.assertEqual(len(generate_inbox(10)), 10)

    def test_inbox_exact_size_15(self):
        self.assertEqual(len(generate_inbox(15)), 15)

    def test_inbox_exact_size_20(self):
        self.assertEqual(len(generate_inbox(20)), 20)

    def test_inbox_returns_tuples(self):
        inbox = generate_inbox(10)
        for item in inbox:
            self.assertEqual(len(item), 2)
            _, priority = item
            self.assertIn(priority, ("urgent", "high", "normal", "low", "spam"))

    def test_email_fields_populated(self):
        email = generate_email("urgent")
        self.assertTrue(email.id)
        self.assertTrue(email.sender)
        self.assertTrue(email.subject)
        self.assertTrue(email.body)
        self.assertTrue(email.timestamp)

    def test_inbox_has_variety(self):
        inbox = generate_inbox(20)
        priorities = {p for _, p in inbox}
        self.assertGreaterEqual(len(priorities), 4)


# ======================================================================
# Task metadata tests
# ======================================================================

class TestTaskMetadata(unittest.TestCase):

    def test_all_three_tasks_exist(self):
        for t in ["task1", "task2", "task3"]:
            self.assertIn(t, TASKS)

    def test_difficulty_progression(self):
        self.assertEqual(TASKS["task1"].difficulty, "easy")
        self.assertEqual(TASKS["task2"].difficulty, "medium")
        self.assertEqual(TASKS["task3"].difficulty, "hard")

    def test_inbox_size_increases(self):
        self.assertLess(TASKS["task1"].inbox_size, TASKS["task2"].inbox_size)
        self.assertLess(TASKS["task2"].inbox_size, TASKS["task3"].inbox_size)

    def test_max_steps_increases(self):
        self.assertLess(TASKS["task1"].max_steps, TASKS["task2"].max_steps)
        self.assertLess(TASKS["task2"].max_steps, TASKS["task3"].max_steps)

    def test_all_graders_callable(self):
        for task in TASKS.values():
            self.assertTrue(callable(task.grader))

    def test_grader_scores_in_range_for_empty(self):
        for task in TASKS.values():
            score = task.grader([], {})
            self.assertGreaterEqual(score, 0.0)
            self.assertLessEqual(score, 1.0)


# ======================================================================
# Flask API tests
# ======================================================================

class TestFlaskAPI(unittest.TestCase):

    def setUp(self):
        from api.server import app
        app.config["TESTING"] = True
        self.client = app.test_client()

    def test_root_returns_200(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data["status"], "ready")

    def test_health_returns_ok(self):
        r = self.client.get("/health")
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.get_json()["status"], "ok")

    def test_reset_task1(self):
        r = self.client.post("/reset", json={"task_id": "task1"})
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertEqual(data["task_id"], "task1")
        self.assertEqual(len(data["inbox"]), 10)

    def test_reset_task2(self):
        r = self.client.post("/reset", json={"task_id": "task2"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.get_json()["inbox"]), 15)

    def test_reset_task3(self):
        r = self.client.post("/reset", json={"task_id": "task3"})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(len(r.get_json()["inbox"]), 20)

    def test_reset_invalid_task(self):
        r = self.client.post("/reset", json={"task_id": "task99"})
        self.assertEqual(r.status_code, 400)

    def test_step_classify(self):
        self.client.post("/reset", json={"task_id": "task1"})
        r_obs = self.client.get("/state")
        obs_data = r_obs.get_json()
        email_id = obs_data["processed"][0]["id"]

        r = self.client.post("/step", json={
            "action_type": "classify",
            "email_id": email_id,
            "priority": "urgent"
        })
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("observation", data)
        self.assertIn("reward", data)
        self.assertIn("done", data)
        self.assertGreaterEqual(data["reward"]["total"], 0.0)
        self.assertLessEqual(data["reward"]["total"], 1.0)

    def test_step_invalid_email(self):
        self.client.post("/reset", json={"task_id": "task1"})
        r = self.client.post("/step", json={
            "action_type": "classify",
            "email_id": "nonexistent_id",
            "priority": "spam"
        })
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("error", data["info"])

    def test_state_endpoint(self):
        self.client.post("/reset", json={"task_id": "task1"})
        r = self.client.get("/state")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("task_id", data)
        self.assertIn("step", data)

    def test_tasks_endpoint(self):
        r = self.client.get("/tasks")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("task1", data)
        self.assertIn("task2", data)
        self.assertIn("task3", data)
        self.assertIn("action_schema", data["task1"])

    def test_grader_endpoint(self):
        self.client.post("/reset", json={"task_id": "task1"})
        r = self.client.post("/grader")
        self.assertEqual(r.status_code, 200)
        data = r.get_json()
        self.assertIn("score", data)
        self.assertGreaterEqual(data["score"], 0.0)
        self.assertLessEqual(data["score"], 1.0)

    def test_grader_before_reset_returns_score(self):
        from api.server import env as api_env
        api_env.reset("task1")
        r = self.client.post("/grader")
        self.assertEqual(r.status_code, 200)

    def test_full_episode_task1(self):
        """Simulate a complete task1 episode via API."""
        r = self.client.post("/reset", json={"task_id": "task1"})
        obs = r.get_json()
        done = False
        for email in obs["inbox"]:
            r = self.client.post("/step", json={
                "action_type": "classify",
                "email_id": email["id"],
                "priority": "normal"
            })
            data = r.get_json()
            done = data["done"]
        self.assertTrue(done)

        r = self.client.post("/grader")
        score = r.get_json()["score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    print("=" * 60)
    print("Email Triage OpenEnv - Full Test Suite")
    print("=" * 60)
    unittest.main(verbosity=2)
