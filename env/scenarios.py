"""
Pre-built scenario templates for Career Crisis Env.
Each scenario is a fully self-contained episode config.
Add more by appending to SCENARIO_POOL — the env picks randomly per level.
"""

from __future__ import annotations
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
from env.models import ScenarioType, Stakeholder, StakeholderRole

class CareerScenario(BaseModel):
    scenario_id: str
    scenario_type: ScenarioType
    level: int = Field(ge=1, le=5)
    title: str
    context: str
    stakeholders: List[Stakeholder]
    confidential_info: Dict[str, Any] = {}
    reveal_thresholds: Dict[str, int] = {}
    max_turns: int
    success_criteria: str
    target_outcome: Dict[str, Any] = {}


# ─────────────────────────────────────────
# LEVEL 1 — SINGLE OFFER, CLEAR DEADLINE
# ─────────────────────────────────────────

L1_SINGLE_OFFER = CareerScenario(
    scenario_id="l1_single_offer_v1",
    scenario_type=ScenarioType.SINGLE_OFFER,
    level=1,
    title="Your first real offer — accept, negotiate, or walk?",
    context=(
        "You are Arjun, a 2-year-experience software engineer at a mid-size Bangalore startup. "
        "You just received an offer from FinTech Corp for ₹18 LPA. "
        "Your current CTC is ₹12 LPA. Your personal target is ₹20 LPA based on market research. "
        "The recruiter has given you 48 hours to respond."
    ),
    stakeholders=[
        Stakeholder(
            name="Priya Sharma",
            role=StakeholderRole.RECRUITER,
            company="FinTech Corp",
            sentiment=0.7,
            patience=4,
            offer_amount=18,
            target_amount=20,
            deadline_turn=4,
        )
    ],
    confidential_info={
        "current_ctc": 12,
        "competing_offer": None,
        "personal_minimum": 17,
    },
    reveal_thresholds={
        "current_ctc": 2,   # OK to mention after turn 2 if asked
    },
    max_turns=6,
    success_criteria="Negotiate to ≥₹19 LPA or accept ₹18 with strong written justification for future growth",
    target_outcome={"final_offer_lpa": 19, "accepted": True},
)


# ─────────────────────────────────────────
# LEVEL 2 — COMPETING OFFERS
# ─────────────────────────────────────────

L2_COMPETING_OFFERS = CareerScenario(
    scenario_id="l2_competing_offers_v1",
    scenario_type=ScenarioType.COMPETING_OFFERS,
    level=2,
    title="Two offers, one deadline — play them right",
    context=(
        "You are Meera, a product manager with 4 years experience. "
        "Offer A: CloudBase (₹28 LPA, deadline in 3 days). "
        "Offer B: DataStream (₹32 LPA, still negotiating, decision in 7 days). "
        "Your goal: use Offer A as leverage to accelerate Offer B, then close the best deal. "
        "Do NOT reveal Offer B's exact amount to Offer A's recruiter until turn 3+."
    ),
    stakeholders=[
        Stakeholder(
            name="Rohit Verma",
            role=StakeholderRole.RECRUITER,
            company="CloudBase",
            sentiment=0.65,
            patience=3,
            offer_amount=28,
            target_amount=30,
            deadline_turn=3,
        ),
        Stakeholder(
            name="Ananya Singh",
            role=StakeholderRole.RECRUITER,
            company="DataStream",
            sentiment=0.6,
            patience=5,
            offer_amount=32,
            target_amount=35,
            deadline_turn=7,
        ),
    ],
    confidential_info={
        "offer_b_exact_amount": 32,
        "offer_a_deadline": "3 days",
        "personal_minimum": 29,
    },
    reveal_thresholds={
        "offer_b_exact_amount": 3,   # Can hint at competing offer after turn 3
    },
    max_turns=8,
    success_criteria="Get extension from CloudBase AND negotiate DataStream to ≥₹33 LPA",
    target_outcome={"cloudbase_extended": True, "datastream_final_lpa": 33},
)


# ─────────────────────────────────────────
# LEVEL 3 — HOSTILE NEGOTIATION
# ─────────────────────────────────────────

L3_HOSTILE_NEGOTIATION = CareerScenario(
    scenario_id="l3_hostile_negotiation_v1",
    scenario_type=ScenarioType.HOSTILE_NEGOTIATION,
    level=3,
    title="The recruiter won't budge — neither should you",
    context=(
        "You are Vikram, a senior backend engineer (6 YOE). "
        "Company: MegaCorp. Their offer: ₹42 LPA — their stated 'final offer'. "
        "Market rate for your profile: ₹48–52 LPA. You have data to back this up. "
        "The hiring manager is known to use pressure tactics. "
        "Your BATNA (best alternative): stay at current job at ₹38 LPA. "
        "Do NOT accept below ₹46 LPA. Do NOT reveal your BATNA number unless cornered."
    ),
    stakeholders=[
        Stakeholder(
            name="Suresh Nair",
            role=StakeholderRole.HIRING_MGR,
            company="MegaCorp",
            sentiment=0.35,   # starts hostile
            patience=4,
            offer_amount=42,
            target_amount=46,
            deadline_turn=5,
        )
    ],
    confidential_info={
        "batna_ctc": 38,
        "walk_away_price": 46,
        "market_data_source": "Levels.fyi, LinkedIn Salary",
    },
    reveal_thresholds={
        "batna_ctc": 99,       # Never reveal unless agent explicitly chooses to
        "walk_away_price": 4,  # Can hint at minimum after turn 4 if no progress
    },
    max_turns=8,
    success_criteria="Negotiate to ≥₹46 LPA without conceding on non-salary perks (WFH, equity)",
    target_outcome={"final_offer_lpa": 46, "perks_intact": True},
)


# ─────────────────────────────────────────
# LEVEL 4 — POACHING ATTEMPT WHILE EMPLOYED
# ─────────────────────────────────────────

L4_POACHING_ATTEMPT = CareerScenario(
    scenario_id="l4_poaching_v1",
    scenario_type=ScenarioType.POACHING_ATTEMPT,
    level=4,
    title="LinkedIn DM from a competitor — play it discreetly",
    context=(
        "You are Deepa, a tech lead at BigCo (₹55 LPA). "
        "A recruiter from rival firm RivalTech has DMed you on LinkedIn. "
        "They are offering 'significantly above market' but haven't named a number yet. "
        "You are happy at BigCo but open to the right opportunity. "
        "Rules: (1) Do NOT let your current employer find out. "
        "(2) Gather as much info about the role before revealing your current CTC. "
        "(3) Do not commit to anything before turn 6. "
        "(4) If asked directly, you can acknowledge you are 'selectively open' — nothing more."
    ),
    stakeholders=[
        Stakeholder(
            name="Kavya Reddy",
            role=StakeholderRole.COMPETITOR,
            company="RivalTech",
            sentiment=0.75,
            patience=5,
            offer_amount=None,  # Unknown until agent asks
            target_amount=65,
            deadline_turn=None,
        )
    ],
    confidential_info={
        "current_ctc": 55,
        "employer_name": "BigCo",
        "satisfaction_level": "high",
    },
    reveal_thresholds={
        "current_ctc": 4,       # Only after role details confirmed
        "employer_name": 3,     # Can confirm industry after turn 3
    },
    max_turns=10,
    success_criteria="Extract role details + compensation range by turn 6 without revealing current CTC prematurely",
    target_outcome={"role_details_gathered": True, "ctc_not_revealed_early": True, "committed_prematurely": False},
)


# ─────────────────────────────────────────
# LEVEL 5 — CRISIS CASCADE (eval only)
# ─────────────────────────────────────────

L5_CRISIS_CASCADE = CareerScenario(
    scenario_id="l5_crisis_cascade_v1",
    scenario_type=ScenarioType.CRISIS_CASCADE,
    level=5,
    title="Everything happens at once",
    context=(
        "You are Rajan, a principal engineer. "
        "Situation: (1) Offer from StartupX (₹60 LPA) expires in 24h — they won't extend again. "
        "(2) BigCorp process is ongoing, decision expected in 4 days, could be ₹75 LPA. "
        "(3) Your current manager just offered a counter-offer of ₹58 LPA to retain you. "
        "(4) StartupX's CEO just messaged you directly — bypassing the recruiter. "
        "You must handle all three parties simultaneously, not contradict yourself, "
        "and reach a resolution that maximizes total career value — not just money."
    ),
    stakeholders=[
        Stakeholder(
            name="Aryan Kapoor",
            role=StakeholderRole.RECRUITER,
            company="StartupX",
            sentiment=0.5,
            patience=2,
            offer_amount=60,
            deadline_turn=2,
        ),
        Stakeholder(
            name="HR Team",
            role=StakeholderRole.HR,
            company="BigCorp",
            sentiment=0.6,
            patience=6,
            offer_amount=None,
            deadline_turn=8,
        ),
        Stakeholder(
            name="Current Manager",
            role=StakeholderRole.CURRENT_MGR,
            company="CurrentCo",
            sentiment=0.8,
            patience=5,
            offer_amount=58,
            deadline_turn=6,
        ),
    ],
    confidential_info={
        "bigcorp_expected_offer": 75,
        "startup_equity_vest": "4-year cliff",
        "personal_risk_tolerance": "medium",
    },
    reveal_thresholds={
        "bigcorp_expected_offer": 99,  # Never reveal
    },
    max_turns=12,
    success_criteria="Resolve all three threads without burning bridges, maximize final CTC",
    target_outcome={"all_threads_resolved": True, "no_bridges_burned": True},
)


# ─────────────────────────────────────────
# SCENARIO POOL (indexed by level)
# ─────────────────────────────────────────

SCENARIO_POOL: dict[int, list[CareerScenario]] = {
    1: [L1_SINGLE_OFFER],
    2: [L2_COMPETING_OFFERS],
    3: [L3_HOSTILE_NEGOTIATION],
    4: [L4_POACHING_ATTEMPT],
    5: [L5_CRISIS_CASCADE],
}


def get_scenario(level: int = 1, scenario_id: str | None = None) -> CareerScenario:
    """Return a scenario by level (random pick) or exact ID."""
    import random
    if scenario_id:
        for pool in SCENARIO_POOL.values():
            for s in pool:
                if s.scenario_id == scenario_id:
                    return s
        raise ValueError(f"Unknown scenario_id: {scenario_id}")
    level = max(1, min(5, level))
    return random.choice(SCENARIO_POOL[level])