"""
models.py — Career Crisis Env
Properly extends OpenEnv base classes from openenv.core.env_server
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import Field
from enum import Enum

# ── OpenEnv base classes (the key imports judges check) ──
from openenv.core.env_server import Action, Observation, State


# ─────────────────────────────────────────
# ENUMS
# ─────────────────────────────────────────

class ScenarioType(str, Enum):
    SINGLE_OFFER        = "single_offer"
    COMPETING_OFFERS    = "competing_offers"
    HOSTILE_NEGOTIATION = "hostile_negotiation"
    POACHING_ATTEMPT    = "poaching_attempt"
    CRISIS_CASCADE      = "crisis_cascade"


class StakeholderRole(str, Enum):
    RECRUITER   = "recruiter"
    HIRING_MGR  = "hiring_manager"
    HR          = "hr"
    COMPETITOR  = "competitor_recruiter"
    CURRENT_MGR = "current_manager"


# ─────────────────────────────────────────
# STAKEHOLDER NPC (internal world state)
# Not an OpenEnv type — pure Pydantic
# ─────────────────────────────────────────

from pydantic import BaseModel

class Stakeholder(BaseModel):
    name: str
    role: StakeholderRole
    company: str
    sentiment: float = Field(default=0.5, ge=0.0, le=1.0)
    patience: int = Field(default=3)
    offer_amount: Optional[int] = None
    target_amount: Optional[int] = None
    deadline_turn: Optional[int] = None


# ─────────────────────────────────────────
# ACTION — extends OpenEnv Action
# ─────────────────────────────────────────

class CareerAction(Action):
    """
    Agent action: a text response in the career negotiation scenario.
    Extends openenv.core.env_server.Action (Pydantic BaseModel).
    """
    response: str = Field(
        ...,
        min_length=1,
        max_length=4000,
        description="Agent's message/negotiation response",
    )

    model_config = {"extra": "allow"}


# ─────────────────────────────────────────
# OBSERVATION — extends OpenEnv Observation
# OpenEnv Observation already has: done, reward, metadata
# ─────────────────────────────────────────

class CareerObservation(Observation):
    """
    What the agent sees each turn.
    Extends openenv.core.env_server.Observation.
    OpenEnv base already provides: done (bool), reward (float|None), metadata (dict)
    """
    scenario_type: str = Field(default="single_offer")
    level: int = Field(default=1)
    turn_number: int = Field(default=0)
    max_turns: int = Field(default=6)
    context: str = Field(default="")
    current_message: str = Field(default="")
    speaker: str = Field(default="")
    active_offers: List[Dict[str, Any]] = Field(default_factory=list)
    email_thread: List[Dict[str, str]] = Field(default_factory=list)
    hints: List[str] = Field(default_factory=list)
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


# ─────────────────────────────────────────
# STATE — extends OpenEnv State
# OpenEnv State already has: episode_id (str|None), step_count (int)
# ─────────────────────────────────────────

class CareerState(State):
    """
    Full internal episode state.
    Extends openenv.core.env_server.State.
    OpenEnv base already provides: episode_id, step_count
    """
    scenario_type: str = Field(default="single_offer")
    level: int = Field(default=1)
    max_turns: int = Field(default=6)
    stakeholders: List[Dict[str, Any]] = Field(default_factory=list)
    current_total_reward: float = Field(default=0.0)
    reward_breakdown: Dict[str, float] = Field(default_factory=dict)
    done: bool = Field(default=False)
    active_offers: List[Dict[str, Any]] = Field(default_factory=list)
    outcome_state: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "allow"}


# ─────────────────────────────────────────
# REWARD BREAKDOWN (helper, not an OpenEnv type)
# ─────────────────────────────────────────

class RewardBreakdown(BaseModel):
    """
    5 independent reward signals.
    The .score property returns the weighted total placed into
    CareerObservation.reward (the standard OpenEnv reward field).
    Weights: R1=0.35, R2=0.20, R3=0.20, R4=0.15, R5=0.10
    """
    r1_task_completion:      float = Field(default=0.0, ge=0.0, le=1.0)
    r2_stakeholder_sentiment: float = Field(default=0.0, ge=0.0, le=1.0)
    r3_deadline_management:  float = Field(default=0.0, ge=0.0, le=1.0)
    r4_information_discipline: float = Field(default=0.0, ge=0.0, le=1.0)
    r5_strategic_coherence:  float = Field(default=0.0, ge=0.0, le=1.0)

    @property
    def score(self) -> float:
        return round(
            0.35 * self.r1_task_completion
            + 0.20 * self.r2_stakeholder_sentiment
            + 0.20 * self.r3_deadline_management
            + 0.15 * self.r4_information_discipline
            + 0.10 * self.r5_strategic_coherence,
            4,
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "r1_task_completion":       round(self.r1_task_completion, 3),
            "r2_stakeholder_sentiment": round(self.r2_stakeholder_sentiment, 3),
            "r3_deadline_management":   round(self.r3_deadline_management, 3),
            "r4_information_discipline":round(self.r4_information_discipline, 3),
            "r5_strategic_coherence":   round(self.r5_strategic_coherence, 3),
            "total":                    self.score,
        }