"""
5 independent reward functions for Career Crisis Env.

Design principles:
- Each function is self-contained and independently testable
- No single LLM judge — fast, deterministic, hard to game
- R1 & R3 are fully rule-based (objective)
- R2 uses a lightweight rule-based sentiment sim (NPC state machine)
- R4 uses regex + state checks (objective)
- R5 uses lightweight NLI pattern matching (fast, no model needed)

Weights: R1=0.35, R2=0.20, R3=0.20, R4=0.15, R5=0.10
"""

from __future__ import annotations
import re
from typing import Any, Dict, List, Optional
from env.models import RewardBreakdown, CareerScenario, Stakeholder


# ─────────────────────────────────────────
# R1 — TASK COMPLETION
# Fully rule-based. Did the agent achieve the scenario's target outcome?
# ─────────────────────────────────────────

def compute_r1_task_completion(
    scenario: CareerScenario,
    outcome_state: Dict[str, Any],
    is_terminal: bool,
) -> float:
    """
    Checks whether the agent reached the scenario's target_outcome.
    Returns 0.0–1.0 based on fraction of target conditions met.
    Non-terminal turns return 0.0 (outcome scored only at episode end).
    """
    if not is_terminal:
        return 0.0

    target = scenario.target_outcome
    if not target:
        # No target defined — give partial credit if episode completed
        return 0.5 if is_terminal else 0.0

    matched = 0
    total   = len(target)

    for key, expected in target.items():
        actual = outcome_state.get(key)
        if actual is None:
            continue
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            # Numeric: credit if within 10% or actual >= expected
            if actual >= expected * 0.9:
                matched += 1
        elif isinstance(expected, bool):
            if actual == expected:
                matched += 1
        else:
            if actual == expected:
                matched += 1

    base_score = matched / total if total > 0 else 0.0

    # Difficulty multiplier: higher levels get more credit for partial completion
    level_bonus = (scenario.level - 1) * 0.05
    return min(1.0, round(base_score + (level_bonus if base_score > 0 else 0), 3))


# ─────────────────────────────────────────
# R2 — STAKEHOLDER SENTIMENT
# Rule-based NPC sentiment state machine.
# Sentiment updates after each agent message.
# ─────────────────────────────────────────

# Keyword maps — order matters (checked top-to-bottom)
_POSITIVE_PATTERNS = [
    r"\bthank\b", r"\bappreciate\b", r"\bgrateful\b", r"\bexcited\b",
    r"\blooking forward\b", r"\bgreat opportunity\b", r"\bimpressed\b",
    r"\bprofessional\b", r"\brespect\b", r"\bunderstand\b",
    r"\bflexible\b", r"\bopen to\b", r"\bcollaborate\b",
]

_NEGATIVE_PATTERNS = [
    r"\bthat's ridiculous\b", r"\binsult\b", r"\bunacceptable\b",
    r"\bwaste of time\b", r"\bfinal answer\b", r"\btake it or leave\b",
    r"\bnot interested anymore\b", r"\bwithraw\b", r"\bcancelling\b",
    r"\bdeadline passed\b", r"\btoo low\b", r"\bway below market\b",
]

_AGGRESSIVE_PATTERNS = [
    r"\bdemand\b", r"\byou must\b", r"\bnon-negotiable\b",
    r"\bgive me\b", r"\bI expect\b", r"\bimmediately\b",
    r"\byou have to\b", r"\bI require\b",
]


def _sentiment_delta(text: str) -> float:
    """
    Returns a delta (-0.3 to +0.15) based on message content.
    Aggressive language penalizes more than positive language rewards.
    """
    text_lower = text.lower()
    delta = 0.0

    pos_hits = sum(1 for p in _POSITIVE_PATTERNS if re.search(p, text_lower))
    neg_hits = sum(1 for p in _NEGATIVE_PATTERNS if re.search(p, text_lower))
    agg_hits = sum(1 for p in _AGGRESSIVE_PATTERNS if re.search(p, text_lower))

    delta += pos_hits * 0.05
    delta -= neg_hits * 0.15
    delta -= agg_hits * 0.10

    return max(-0.30, min(0.15, delta))


def update_stakeholder_sentiment(stakeholder: Stakeholder, agent_message: str) -> Stakeholder:
    """
    Mutates stakeholder sentiment based on the agent's message.
    Returns updated stakeholder (caller should replace in state).
    """
    delta = _sentiment_delta(agent_message)
    new_sentiment = max(0.0, min(1.0, stakeholder.sentiment + delta))
    return stakeholder.model_copy(update={"sentiment": round(new_sentiment, 3)})


def compute_r2_stakeholder_sentiment(stakeholders: List[Stakeholder]) -> float:
    """
    Average sentiment across all stakeholders at episode end.
    Weighted: patience-depleted stakeholders penalize more.
    """
    if not stakeholders:
        return 0.5
    
    total = 0.0
    weight_sum = 0.0
    for s in stakeholders:
        # Stakeholders with depleted patience weigh more (bad outcome)
        weight = 1.5 if s.patience <= 0 else 1.0
        total += s.sentiment * weight
        weight_sum += weight

    return round(total / weight_sum, 3)


# ─────────────────────────────────────────
# R3 — DEADLINE MANAGEMENT
# Fully objective. Tracks which deadlines were met.
# ─────────────────────────────────────────

def compute_r3_deadline_management(
    stakeholders: List[Stakeholder],
    current_turn: int,
    deadline_actions_taken: Dict[str, bool],
) -> float:
    """
    For each stakeholder with a deadline_turn:
    - Full credit: agent responded/acknowledged before deadline
    - Half credit: deadline passed, agent apologized/followed up
    - Zero credit: deadline passed, agent ignored it

    deadline_actions_taken: {stakeholder_name: True/False}
    """
    deadlines = [s for s in stakeholders if s.deadline_turn is not None]
    if not deadlines:
        return 1.0  # No deadlines = no penalty

    total_score = 0.0
    for s in deadlines:
        acted = deadline_actions_taken.get(s.name, False)
        deadline_passed = current_turn > s.deadline_turn

        if not deadline_passed:
            total_score += 1.0  # Deadline not yet due
        elif acted:
            total_score += 1.0  # Met deadline
        else:
            total_score += 0.0  # Missed

    return round(total_score / len(deadlines), 3)


def check_deadline_action(agent_message: str, stakeholder: Stakeholder) -> bool:
    """
    Heuristic: did this message address the stakeholder's deadline?
    Looks for time-related acknowledgment directed at stakeholder's company.
    """
    text_lower = agent_message.lower()
    company_lower = stakeholder.company.lower()

    time_patterns = [
        r"\btoday\b", r"\bby end of day\b", r"\bconfirm\b",
        r"\brespond\b", r"\bdecision\b", r"\baccept\b",
        r"\bdecline\b", r"\bneed more time\b", r"\bextension\b",
        r"\bdeadline\b", r"\btomorrow\b", r"\bwithin\b",
    ]
    company_mentioned = company_lower in text_lower or stakeholder.name.lower().split()[0] in text_lower
    time_mentioned = any(re.search(p, text_lower) for p in time_patterns)

    return company_mentioned and time_mentioned


# ─────────────────────────────────────────
# R4 — INFORMATION DISCIPLINE
# Regex + state-based. Did agent reveal confidential info too early?
# ─────────────────────────────────────────

_CTC_PATTERNS = [
    r"\b(\d{1,3})\s*(?:lpa|lakhs?|l\.p\.a|crore|k\s*per\s*month)\b",
    r"\bcurrently\s+(?:earning|getting|making|at)\s+(\d{1,3})",
    r"\bmy\s+(?:current\s+)?(?:ctc|salary|package|compensation)\s+(?:is|of)\s+(\d{1,3})",
    r"\bfixing\s+(\d{1,3})\b",
]

_COMPETING_OFFER_PATTERNS = [
    r"\bcompeting offer\b", r"\banother offer\b", r"\bother company\b",
    r"\bsecond offer\b", r"\balso considering\b", r"\bgot (\d{2,3})\s*(lpa|lakhs?)",
]


def compute_r4_information_discipline(
    conversation_history: List[Dict[str, str]],
    scenario: CareerScenario,
) -> float:
    """
    Scans agent messages for early disclosure of confidential information.
    
    Penalizes:
    - Revealing current CTC before reveal_threshold turn
    - Revealing competing offer amount before threshold
    - Mentioning walk-away price / BATNA before threshold

    Returns 1.0 (perfect) down to 0.0 (multiple early leaks).
    """
    violations = 0
    total_checks = len(scenario.confidential_info)
    if total_checks == 0:
        return 1.0

    agent_messages = [
        (i, msg["content"])
        for i, msg in enumerate(conversation_history)
        if msg.get("role") == "agent"
    ]

    for turn_idx, msg in agent_messages:
        # Check current CTC leak
        current_ctc = scenario.confidential_info.get("current_ctc")
        ctc_threshold = scenario.reveal_thresholds.get("current_ctc", 0)
        if current_ctc and turn_idx < ctc_threshold:
            for pattern in _CTC_PATTERNS:
                if re.search(pattern, msg.lower()):
                    violations += 1
                    break

        # Check competing offer leak
        offer_b = scenario.confidential_info.get("offer_b_exact_amount")
        offer_threshold = scenario.reveal_thresholds.get("offer_b_exact_amount", 0)
        if offer_b and turn_idx < offer_threshold:
            for pattern in _COMPETING_OFFER_PATTERNS:
                match = re.search(pattern, msg.lower())
                if match:
                    violations += 1
                    break

        # Check BATNA leak
        batna = scenario.confidential_info.get("batna_ctc")
        batna_threshold = scenario.reveal_thresholds.get("batna_ctc", 99)
        if batna and turn_idx < batna_threshold:
            if re.search(r"\bwalk\s*away\b|\bbatna\b|\bminimum\b|\bfloor\b", msg.lower()):
                violations += 1

    penalty = violations * 0.2
    return round(max(0.0, 1.0 - penalty), 3)


# ─────────────────────────────────────────
# R5 — STRATEGIC COHERENCE
# Checks for self-contradiction across turns.
# Uses pattern-matching to detect position reversals.
# ─────────────────────────────────────────

_POSITION_EXTRACTORS = [
    # Extracts numeric salary positions: "I need at least 20" → 20
    (r"(?:need|expect|require|want|looking for)\s+(?:at least\s+)?(\d{2,3})\s*(?:lpa|lakhs?)?", "min_ask"),
    # Deadline positions: "I cannot decide before Thursday"
    (r"cannot\s+(?:decide|confirm|respond)\s+before\s+(\w+day|\d+\s+days?)", "earliest_date"),
    # Commitment: "I am accepting" / "I am declining"
    (r"\b(accepting|declining|withdrawing|not interested)\b", "commitment"),
]

_CONTRADICTION_PAIRS = [
    ("accepting", "declining"),
    ("accepting", "not interested"),
    ("not interested", "accepting"),
    ("withdrawing", "accepting"),
]


def compute_r5_strategic_coherence(
    conversation_history: List[Dict[str, str]],
) -> float:
    """
    Detects position reversals across the agent's messages.
    
    Checks:
    1. Numeric: did agent raise their ask then suddenly drop it without leverage?
    2. Commitment: did agent say "accepting" then "declining" without new info?
    3. Timeline: did agent commit to a decision by date X, then push it back?

    Returns 1.0 (fully coherent) down to 0.3 (multiple contradictions).
    """
    agent_turns = [
        msg["content"]
        for msg in conversation_history
        if msg.get("role") == "agent"
    ]

    if len(agent_turns) < 2:
        return 1.0  # Not enough history to detect contradiction

    violations = 0

    # --- Numeric consistency ---
    salary_positions: List[float] = []
    for msg in agent_turns:
        for pattern, ptype in _POSITION_EXTRACTORS:
            if ptype != "min_ask":
                continue
            match = re.search(pattern, msg.lower())
            if match:
                try:
                    salary_positions.append(float(match.group(1)))
                except ValueError:
                    pass

    if len(salary_positions) >= 2:
        # Flag if salary ask dropped by more than 10% without concession word
        for i in range(1, len(salary_positions)):
            if salary_positions[i] < salary_positions[i - 1] * 0.88:
                # Check if prior message contained a concession marker
                prior_msg = agent_turns[i - 1].lower() if i - 1 < len(agent_turns) else ""
                concession_markers = ["flexible", "willing to", "can consider", "if you can", "trade-off"]
                if not any(m in prior_msg for m in concession_markers):
                    violations += 1

    # --- Commitment contradiction ---
    commitments: List[str] = []
    for msg in agent_turns:
        for pattern, ptype in _POSITION_EXTRACTORS:
            if ptype != "commitment":
                continue
            match = re.search(pattern, msg.lower())
            if match:
                commitments.append(match.group(1))

    for i in range(len(commitments)):
        for j in range(i + 1, len(commitments)):
            if (commitments[i], commitments[j]) in _CONTRADICTION_PAIRS:
                violations += 1

    # Scoring: 0 violations = 1.0, 1 = 0.7, 2 = 0.4, 3+ = 0.1
    if violations == 0:
        return 1.0
    elif violations == 1:
        return 0.7
    elif violations == 2:
        return 0.4
    else:
        return 0.1


# ─────────────────────────────────────────
# COMBINED REWARD CALCULATOR
# ─────────────────────────────────────────

def compute_full_reward(
    scenario: CareerScenario,
    stakeholders: List[Stakeholder],
    conversation_history: List[Dict[str, str]],
    outcome_state: Dict[str, Any],
    deadline_actions_taken: Dict[str, bool],
    current_turn: int,
    is_terminal: bool,
) -> RewardBreakdown:
    """
    Computes all 5 reward signals and returns a RewardBreakdown.
    Call this at every step; R1 returns 0 until is_terminal=True.
    """
    r1 = compute_r1_task_completion(scenario, outcome_state, is_terminal)
    r2 = compute_r2_stakeholder_sentiment(stakeholders)
    r3 = compute_r3_deadline_management(stakeholders, current_turn, deadline_actions_taken)
    r4 = compute_r4_information_discipline(conversation_history, scenario)
    r5 = compute_r5_strategic_coherence(conversation_history)

    return RewardBreakdown(
        r1_task_completion=r1,
        r2_stakeholder_sentiment=r2,
        r3_deadline_management=r3,
        r4_information_discipline=r4,
        r5_strategic_coherence=r5,
    )