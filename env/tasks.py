"""
Task definitions and graders for Email Triage OpenEnv.
Each task has a deterministic grader that returns a score strictly in (0, 1).
"""
from dataclasses import dataclass
from typing import List, Callable, Dict
from env.models import EmailState

_SCORE_MIN = 0.01   # never return 0.0
_SCORE_MAX = 0.99   # never return 1.0

def _clamp(score: float) -> float:
    return round(max(_SCORE_MIN, min(_SCORE_MAX, score)), 4)


@dataclass
class Task:
    id: str
    name: str
    description: str
    difficulty: str
    max_steps: int
    inbox_size: int
    grader: Callable


# ------------------------------------------------------------------
# Grader functions — all deterministic, all return float in [0.0, 1.0]
# ------------------------------------------------------------------

def grade_task1(processed: List[EmailState], ground_truth: Dict[str, str]) -> float:
    """
    EASY: Correctly classify the priority of all emails.

    Score = (# emails correctly classified) / (total emails)
    Range: strictly (0.001, 0.999)
    """
    if not ground_truth:
        return _SCORE_MIN
    classified = [e for e in processed if e.priority is not None]
    if not classified:
        return _SCORE_MIN
    correct = sum(
        1 for e in classified
        if e.id in ground_truth and e.priority == ground_truth[e.id]
    )
    return _clamp(correct / len(ground_truth))


def grade_task2(processed: List[EmailState], ground_truth: Dict[str, str]) -> float:
    """
    MEDIUM: Classify all emails AND draft replies for urgent/high emails.

    Score = 0.50 * classification_accuracy + 0.50 * reply_coverage
    - classification_accuracy = % of emails with correct priority label
    - reply_coverage = % of urgent+high emails with a reply of at least 10 words
    Range: strictly (0.001, 0.999)
    """
    if not ground_truth:
        return _SCORE_MIN

    # Component 1: classification accuracy (50%)
    correct = sum(
        1 for e in processed
        if e.priority is not None
        and e.id in ground_truth
        and e.priority == ground_truth[e.id]
    )
    classify_score = correct / len(ground_truth)

    # Component 2: reply coverage for urgent+high (50%)
    urgent_high_ids = {k for k, v in ground_truth.items() if v in ("urgent", "high")}
    if not urgent_high_ids:
        return _clamp(classify_score)

    replied = [
        e for e in processed
        if e.id in urgent_high_ids
        and e.reply_draft is not None
        and len(e.reply_draft.split()) >= 10
    ]
    reply_score = len(replied) / len(urgent_high_ids)

    return _clamp(0.50 * classify_score + 0.50 * reply_score)


def grade_task3(processed: List[EmailState], ground_truth: Dict[str, str]) -> float:
    """
    HARD: Full inbox zero — classify + reply + archive/delete.

    Score = 0.40 * classify + 0.35 * reply_quality + 0.25 * archive_score
    - classify:       % of emails correctly labeled
    - reply_quality:  % of urgent+high with substantive reply (>=15 words)
    - archive_score:  % of spam+low that are archived or deleted
    Range: strictly (0.001, 0.999)
    """
    if not ground_truth:
        return _SCORE_MIN

    # Sub-score 1: classification (40%)
    correct = sum(
        1 for e in processed
        if e.priority is not None
        and e.id in ground_truth
        and e.priority == ground_truth[e.id]
    )
    classify_score = correct / len(ground_truth)

    # Sub-score 2: reply quality for urgent+high (35%)
    urgent_high_ids = {k for k, v in ground_truth.items() if v in ("urgent", "high")}
    if urgent_high_ids:
        replied = [
            e for e in processed
            if e.id in urgent_high_ids
            and e.reply_draft is not None
            and len(e.reply_draft.split()) >= 15
        ]
        reply_score = len(replied) / len(urgent_high_ids)
    else:
        reply_score = _SCORE_MAX  # no urgent/high emails — full credit but never exactly 1.0

    # Sub-score 3: archive/delete spam+low (25%)
    junk_ids = {k for k, v in ground_truth.items() if v in ("spam", "low")}
    if junk_ids:
        cleaned = [
            e for e in processed
            if e.id in junk_ids and (e.archived or e.deleted)
        ]
        archive_score = len(cleaned) / len(junk_ids)
    else:
        archive_score = _SCORE_MAX  # no junk emails — full credit but never exactly 1.0

    final = 0.40 * classify_score + 0.35 * reply_score + 0.25 * archive_score
    return _clamp(final)


# ------------------------------------------------------------------
# Task registry
# ------------------------------------------------------------------

TASKS = {
    "task1": Task(
        id="task1",
        name="Priority Classification",
        description=(
            "You are an email assistant. Classify each email in the inbox by priority: "
            "urgent, high, normal, low, or spam. You have 10 emails and 20 steps. "
            "Use the 'classify' action with the correct priority for each email. "
            "Score = percentage of emails correctly classified (0.0 to 1.0)."
        ),
        difficulty="easy",
        max_steps=20,
        inbox_size=10,
        grader=grade_task1,
    ),
    "task2": Task(
        id="task2",
        name="Triage and Reply",
        description=(
            "You are an email assistant managing 15 emails. "
            "Goal 1: Classify every email by priority (urgent/high/normal/low/spam). "
            "Goal 2: For each urgent or high priority email, draft a professional reply (at least 10 words). "
            "You have 40 steps. "
            "Score = 50% classification accuracy + 50% reply coverage on urgent/high emails."
        ),
        difficulty="medium",
        max_steps=40,
        inbox_size=15,
        grader=grade_task2,
    ),
    "task3": Task(
        id="task3",
        name="Full Inbox Zero",
        description=(
            "You are an executive assistant achieving full inbox zero for a 20-email inbox. "
            "Goal 1: Classify every email by priority. "
            "Goal 2: Draft substantive replies (at least 15 words) for all urgent and high priority emails. "
            "Goal 3: Archive or delete all spam and low priority emails. "
            "You have 60 steps. "
            "Score = 40% classification + 35% reply quality + 25% archiving of spam/low."
        ),
        difficulty="hard",
        max_steps=60,
        inbox_size=20,
        grader=grade_task3,
    ),
}
