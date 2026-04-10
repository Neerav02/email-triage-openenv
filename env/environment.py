"""
Core EmailTriageEnv environment.
Implements OpenEnv spec: reset() / step() / state() / grade()
No external dependencies - stdlib only.
"""
from typing import Tuple, Dict, Any, Optional, List
from env.models import Observation, Action, Reward, EmailState, Email
from env.tasks import TASKS
from env.data_generator import generate_inbox


class EmailTriageEnv:
    """
    Email Triage OpenEnv Environment.

    Simulates real-world email inbox management. Agents classify emails,
    draft replies for urgent messages, and archive junk to achieve inbox zero.

    OpenEnv API:
        obs              = env.reset(task_id)
        obs, rew, done, info = env.step(action)
        snapshot         = env.state()
        score            = env.grade()
    """

    def __init__(self):
        self.task_id: Optional[str] = None
        self.inbox: List[Email] = []
        self.ground_truth: Dict[str, str] = {}      # email_id -> true priority string
        self.processed: Dict[str, EmailState] = {}  # email_id -> agent's EmailState
        self.step_count: int = 0
        self.done: bool = False
        self._cumulative_reward: float = 0.0
        self._action_history: List[Dict] = []

    # ------------------------------------------------------------------
    # OpenEnv core API
    # ------------------------------------------------------------------

    def reset(self, task_id: str = "task1") -> Observation:
        """
        Start a fresh episode for the given task.
        Returns the initial Observation.
        """
        if task_id not in TASKS:
            raise ValueError(
                f"Unknown task '{task_id}'. Valid tasks: {list(TASKS.keys())}"
            )

        task = TASKS[task_id]
        self.task_id = task_id

        raw_inbox = generate_inbox(task.inbox_size)
        self.inbox = [email for email, _ in raw_inbox]
        self.ground_truth = {email.id: priority for email, priority in raw_inbox}
        self.processed = {email.id: EmailState(id=email.id) for email in self.inbox}
        self.step_count = 0
        self.done = False
        self._cumulative_reward = 0.0
        self._action_history = []

        return self._build_observation()

    def step(self, action: Action) -> Tuple[Observation, Reward, bool, Dict]:
        """
        Execute one action. Returns (observation, reward, done, info).

        Action fields:
            action_type : classify | reply | archive | delete | read | skip
            email_id    : target email id
            priority    : urgent|high|normal|low|spam  (for classify)
            reply_text  : string (for reply)
        """
        if self.done:
            raise RuntimeError("Episode is done. Call reset() to start a new episode.")
        if self.task_id is None:
            raise RuntimeError("Not initialized. Call reset() first.")

        task = TASKS[self.task_id]
        self.step_count += 1
        components: Dict[str, float] = {}

        # --- Validate email_id ---
        if action.email_id not in self.processed:
            components["invalid_action_penalty"] = -0.05
            reward = Reward(
                total=max(0.0, self._cumulative_reward),
                components=components,
                done=False,
                info={"error": f"Unknown email_id '{action.email_id}'",
                      "step": self.step_count},
            )
            return self._build_observation(), reward, False, reward.info

        state = self.processed[action.email_id]
        gt = self.ground_truth.get(action.email_id, "normal")

        # --- Apply action ---
        if action.action_type == "classify":
            if not action.priority:
                components["missing_priority"] = -0.02
            else:
                state.priority = action.priority
                if action.label:
                    state.label = action.label
                if action.priority == gt:
                    components["correct_classification"] = 0.10
                elif _adjacent(action.priority, gt):
                    components["near_miss"] = 0.03
                else:
                    components["wrong_classification"] = -0.05

        elif action.action_type == "reply":
            state.reply_draft = action.reply_text or ""
            wc = len(state.reply_draft.split()) if state.reply_draft else 0
            if gt in ("urgent", "high"):
                if wc >= 15:
                    components["quality_reply"] = 0.15
                elif wc >= 10:
                    components["short_reply"] = 0.08
                elif wc > 0:
                    components["minimal_reply"] = 0.02
                else:
                    components["empty_reply"] = -0.03
            elif gt == "normal":
                components["optional_reply"] = 0.03 if wc >= 10 else 0.0
            else:
                components["unnecessary_reply"] = -0.02

        elif action.action_type == "archive":
            if not state.archived:
                state.archived = True
                if gt in ("spam", "low"):
                    components["correct_archive"] = 0.08
                elif gt == "normal":
                    components["questionable_archive"] = -0.01
                else:
                    components["wrong_archive"] = -0.06

        elif action.action_type == "delete":
            if not state.deleted:
                state.deleted = True
                if gt == "spam":
                    components["correct_delete"] = 0.10
                elif gt == "low":
                    components["ok_delete"] = 0.05
                else:
                    components["wrong_delete"] = -0.08

        elif action.action_type in ("read", "skip"):
            components["neutral"] = 0.0

        else:
            components["unknown_action"] = -0.03

        # Step cost to discourage padding/loops
        components["step_cost"] = -0.005

        # Update cumulative
        delta = sum(components.values())
        self._cumulative_reward = max(0.001, min(0.999, self._cumulative_reward + delta))

        # Record history
        self._action_history.append({
            "step": self.step_count,
            "action_type": action.action_type,
            "email_id": action.email_id,
            "delta": round(delta, 4),
        })

        # --- Check termination ---
        max_steps_reached = self.step_count >= task.max_steps
        all_classified = all(e.priority is not None for e in self.processed.values())

        if self.task_id == "task1":
            self.done = max_steps_reached or all_classified
        else:
            self.done = max_steps_reached

        # Final score from grader on episode end
        final_score = 0.0
        if self.done:
            final_score = task.grader(list(self.processed.values()), self.ground_truth)
            components["final_grader_score"] = final_score

        total = final_score if self.done else round(self._cumulative_reward, 4)
        total = max(0.001, min(0.999, total))

        reward = Reward(
            total=total,
            components=components,
            done=self.done,
            info={
                "step": self.step_count,
                "max_steps": task.max_steps,
                "emails_classified": sum(1 for e in self.processed.values() if e.priority),
                "final_score": round(final_score, 4) if self.done else None,
                "task_id": self.task_id,
            },
        )
        return self._build_observation(), reward, self.done, reward.info

    def state(self) -> Dict[str, Any]:
        """Return a snapshot of current episode state."""
        classified = sum(1 for e in self.processed.values() if e.priority)
        replied = sum(
            1 for e in self.processed.values()
            if e.reply_draft and len(e.reply_draft.split()) >= 10
        )
        archived = sum(1 for e in self.processed.values() if e.archived or e.deleted)
        return {
            "task_id": self.task_id,
            "step": self.step_count,
            "done": self.done,
            "inbox_size": len(self.inbox),
            "emails_classified": classified,
            "emails_replied": replied,
            "emails_archived": archived,
            "cumulative_reward": round(self._cumulative_reward, 4),
            "processed": [e.to_dict() for e in self.processed.values()],
            "action_history": self._action_history[-10:],
        }

    def grade(self) -> float:
        """Run the grader on current state. Returns score strictly in (0.001, 0.999)."""
        if self.task_id is None:
            return 0.001
        task = TASKS[self.task_id]
        score = task.grader(list(self.processed.values()), self.ground_truth)
        return round(max(0.001, min(0.999, score)), 4)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_observation(self) -> Observation:
        task = TASKS[self.task_id]
        unclassified = [e for e in self.inbox if self.processed[e.id].priority is None]
        current = unclassified[0] if unclassified else (self.inbox[0] if self.inbox else None)
        return Observation(
            inbox=self.inbox,
            processed=list(self.processed.values()),
            current_email=current,
            step_number=self.step_count,
            max_steps=task.max_steps,
            task_id=self.task_id,
            task_description=task.description,
            available_actions=["classify", "reply", "archive", "delete", "read", "skip"],
        )


def _adjacent(p1: str, p2: str) -> bool:
    """True if p1 and p2 are one priority level apart."""
    order = ["urgent", "high", "normal", "low", "spam"]
    if p1 not in order or p2 not in order:
        return False
    return abs(order.index(p1) - order.index(p2)) == 1
