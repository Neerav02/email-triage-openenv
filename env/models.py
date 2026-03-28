"""
Typed models for Email Triage OpenEnv.
Uses Python dataclasses (stdlib only - no external dependencies).
"""
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Dict, Any
from enum import Enum


class Priority(str, Enum):
    URGENT = "urgent"
    HIGH   = "high"
    NORMAL = "normal"
    LOW    = "low"
    SPAM   = "spam"


@dataclass
class Email:
    id: str
    sender: str
    subject: str
    body: str
    timestamp: str
    thread_id: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class EmailState:
    id: str
    priority: Optional[str] = None
    label: Optional[str] = None
    reply_draft: Optional[str] = None
    archived: bool = False
    deleted: bool = False

    def to_dict(self):
        return asdict(self)


@dataclass
class Observation:
    inbox: List[Email]
    processed: List[EmailState]
    current_email: Optional[Email]
    step_number: int
    max_steps: int
    task_id: str
    task_description: str
    available_actions: List[str]

    def to_dict(self):
        return {
            "inbox": [e.to_dict() for e in self.inbox],
            "processed": [e.to_dict() for e in self.processed],
            "current_email": self.current_email.to_dict() if self.current_email else None,
            "step_number": self.step_number,
            "max_steps": self.max_steps,
            "task_id": self.task_id,
            "task_description": self.task_description,
            "available_actions": self.available_actions,
        }


@dataclass
class Action:
    action_type: str          # classify | reply | archive | delete | read | skip
    email_id: str
    priority: Optional[str] = None
    label: Optional[str] = None
    reply_text: Optional[str] = None

    @classmethod
    def from_dict(cls, d: dict) -> "Action":
        return cls(
            action_type=d.get("action_type", ""),
            email_id=d.get("email_id", ""),
            priority=d.get("priority"),
            label=d.get("label"),
            reply_text=d.get("reply_text"),
        )

    def to_dict(self):
        return asdict(self)


@dataclass
class Reward:
    total: float
    components: Dict[str, float]
    done: bool
    info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self):
        return {
            "total": round(self.total, 4),
            "components": {k: round(v, 4) for k, v in self.components.items()},
            "done": self.done,
            "info": self.info,
        }


# JSON schema for /tasks endpoint (documents action_type options)
ACTION_SCHEMA = {
    "type": "object",
    "required": ["action_type", "email_id"],
    "properties": {
        "action_type": {
            "type": "string",
            "enum": ["classify", "reply", "archive", "delete", "read", "skip"],
            "description": "The action to perform on the email"
        },
        "email_id": {
            "type": "string",
            "description": "ID of the target email from the inbox"
        },
        "priority": {
            "type": "string",
            "enum": ["urgent", "high", "normal", "low", "spam"],
            "description": "Required when action_type is 'classify'"
        },
        "reply_text": {
            "type": "string",
            "description": "Reply draft text. Required when action_type is 'reply'. Min 10 words."
        },
        "label": {
            "type": "string",
            "description": "Optional custom tag for the email"
        }
    }
}
