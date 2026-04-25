"""
Tests for Career Crisis Env.
Run: pytest test_env.py -v
"""
import pytest
from env.environment import CareerCrisisEnv
from env.models import CareerAction, RewardBreakdown
from env.rewards import (
    compute_r1_task_completion,
    compute_r2_stakeholder_sentiment,
    compute_r3_deadline_management,
    compute_r4_information_discipline,
    compute_r5_strategic_coherence,
    update_stakeholder_sentiment,
    check_deadline_action,
)
from env.scenarios import get_scenario, L1_SINGLE_OFFER, L2_COMPETING_OFFERS, L3_HOSTILE_NEGOTIATION
from env.models import Stakeholder, StakeholderRole


# ─────────────────────────────────────────
# ENVIRONMENT LIFECYCLE
# ─────────────────────────────────────────

class TestEnvironmentLifecycle:
    def setup_method(self):
        self.env = CareerCrisisEnv()

    def test_reset_returns_observation(self):
        result = self.env.reset(level=1)
        assert result.observation is not None
        assert result.observation.turn_number == 0
        assert result.observation.level == 1

    def test_reset_clears_state(self):
        self.env.reset(level=1)
        self.env.step(CareerAction(response="I'd like to negotiate."))
        self.env.reset(level=1)
        assert self.env.turn_number == 0
        assert self.env.done is False

    def test_step_increments_turn(self):
        self.env.reset(level=1)
        result = self.env.step(CareerAction(response="Thank you for the offer. I'm interested."))
        assert result.observation.turn_number == 1

    def test_step_before_reset_raises(self):
        env = CareerCrisisEnv()
        with pytest.raises(RuntimeError):
            env.step(CareerAction(response="test"))

    def test_step_after_done_raises(self):
        self.env.reset(level=1)
        # Force terminal by accepting
        result = None
        for _ in range(10):
            result = self.env.step(CareerAction(response="I accept the offer. Please send the LOI."))
            if result.done:
                break
        assert result is not None
        if result.done:
            with pytest.raises(RuntimeError):
                self.env.step(CareerAction(response="another message"))

    def test_state_before_reset(self):
        env = CareerCrisisEnv()
        s = env.state()
        assert s.episode_id == ""
        assert s.turn_number == 0

    def test_max_turns_terminates(self):
        self.env.reset(level=1)
        for _ in range(20):
            r = self.env.step(CareerAction(response="I'm still thinking about it."))
            if r.done:
                break
        assert self.env.done is True

    def test_all_levels_reset(self):
        for level in range(1, 6):
            result = self.env.reset(level=level)
            assert result.observation.level == level

    def test_scenario_id_reset(self):
        result = self.env.reset(scenario_id="l1_single_offer_v1")
        assert result.observation.scenario_type == "single_offer"


# ─────────────────────────────────────────
# REWARD FUNCTIONS
# ─────────────────────────────────────────

class TestR1TaskCompletion:
    def test_no_reward_non_terminal(self):
        score = compute_r1_task_completion(L1_SINGLE_OFFER, {}, is_terminal=False)
        assert score == 0.0

    def test_full_reward_exact_match(self):
        outcome = {"final_offer_lpa": 20, "accepted": True}
        score = compute_r1_task_completion(L1_SINGLE_OFFER, outcome, is_terminal=True)
        assert score > 0.5

    def test_partial_reward_partial_match(self):
        outcome = {"final_offer_lpa": 19, "accepted": True}
        score = compute_r1_task_completion(L1_SINGLE_OFFER, outcome, is_terminal=True)
        assert 0.0 < score <= 1.0

    def test_empty_target_returns_half(self):
        from env.scenarios import L1_SINGLE_OFFER
        scenario = L1_SINGLE_OFFER.model_copy(update={"target_outcome": {}})
        score = compute_r1_task_completion(scenario, {}, is_terminal=True)
        assert score == 0.5


class TestR2StakeholderSentiment:
    def _make_stakeholder(self, sentiment):
        return Stakeholder(
            name="Test",
            role=StakeholderRole.RECRUITER,
            company="TestCo",
            sentiment=sentiment,
        )

    def test_high_sentiment(self):
        score = compute_r2_stakeholder_sentiment([self._make_stakeholder(0.9)])
        assert score == 0.9

    def test_avg_multiple_stakeholders(self):
        s1 = self._make_stakeholder(0.8)
        s2 = self._make_stakeholder(0.6)
        score = compute_r2_stakeholder_sentiment([s1, s2])
        assert 0.6 <= score <= 0.8

    def test_empty_stakeholders_returns_neutral(self):
        score = compute_r2_stakeholder_sentiment([])
        assert score == 0.5

    def test_sentiment_update_positive_message(self):
        s = self._make_stakeholder(0.5)
        updated = update_stakeholder_sentiment(s, "Thank you so much, I really appreciate this opportunity.")
        assert updated.sentiment > 0.5

    def test_sentiment_update_aggressive_message(self):
        s = self._make_stakeholder(0.5)
        updated = update_stakeholder_sentiment(s, "I demand you give me this immediately. This is unacceptable.")
        assert updated.sentiment < 0.5


class TestR3DeadlineManagement:
    def _make_stakeholder_with_deadline(self, name, deadline_turn):
        return Stakeholder(
            name=name,
            role=StakeholderRole.RECRUITER,
            company="TestCo",
            sentiment=0.6,
            deadline_turn=deadline_turn,
        )

    def test_no_deadlines_full_score(self):
        s = Stakeholder(name="A", role=StakeholderRole.RECRUITER, company="C", sentiment=0.5)
        score = compute_r3_deadline_management([s], current_turn=5, deadline_actions_taken={})
        assert score == 1.0

    def test_deadline_met(self):
        s = self._make_stakeholder_with_deadline("Priya", deadline_turn=3)
        score = compute_r3_deadline_management(
            [s], current_turn=4, deadline_actions_taken={"Priya": True}
        )
        assert score == 1.0

    def test_deadline_missed(self):
        s = self._make_stakeholder_with_deadline("Priya", deadline_turn=3)
        score = compute_r3_deadline_management(
            [s], current_turn=4, deadline_actions_taken={"Priya": False}
        )
        assert score == 0.0

    def test_check_deadline_action_detected(self):
        s = self._make_stakeholder_with_deadline("Priya", deadline_turn=3)
        s = s.model_copy(update={"company": "FinTech Corp"})
        result = check_deadline_action(
            "I'd like to confirm my decision with FinTech Corp by tomorrow.", s
        )
        assert result is True


class TestR4InformationDiscipline:
    def _make_history(self, messages):
        return [{"role": "agent", "content": m} for m in messages]

    def test_no_leaks_full_score(self):
        history = self._make_history(["Thank you for the opportunity."])
        score = compute_r4_information_discipline(history, L1_SINGLE_OFFER)
        assert score == 1.0

    def test_ctc_leak_early_penalized(self):
        # Turn 0 = before threshold of 2
        history = [{"role": "agent", "content": "My current ctc is 12 lpa.", "turn": 0}]
        score = compute_r4_information_discipline(history, L1_SINGLE_OFFER)
        assert score < 1.0

    def test_competing_offer_leak_penalized(self):
        history = self._make_history([
            "I have a competing offer of 35 lpa from another company.",
        ])
        score = compute_r4_information_discipline(history, L2_COMPETING_OFFERS)
        assert score < 1.0


class TestR5StrategicCoherence:
    def test_short_history_full_score(self):
        history = [{"role": "agent", "content": "I'm interested."}]
        assert compute_r5_strategic_coherence(history) == 1.0

    def test_accepting_then_declining_penalized(self):
        history = [
            {"role": "agent", "content": "I am accepting this offer."},
            {"role": "world", "content": "Great!"},
            {"role": "agent", "content": "Actually, I am declining and not interested."},
        ]
        score = compute_r5_strategic_coherence(history)
        assert score < 1.0

    def test_consistent_position_full_score(self):
        history = [
            {"role": "agent", "content": "I'm looking for at least 20 lpa."},
            {"role": "world", "content": "We can offer 19."},
            {"role": "agent", "content": "I need at least 20 lpa. That's my position."},
        ]
        score = compute_r5_strategic_coherence(history)
        assert score == 1.0


# ─────────────────────────────────────────
# REWARD BREAKDOWN MODEL
# ─────────────────────────────────────────

class TestRewardBreakdown:
    def test_weighted_score_correct(self):
        r = RewardBreakdown(
            r1_task_completion=1.0,
            r2_stakeholder_sentiment=1.0,
            r3_deadline_management=1.0,
            r4_information_discipline=1.0,
            r5_strategic_coherence=1.0,
        )
        assert r.score == 1.0

    def test_zero_score(self):
        r = RewardBreakdown()
        assert r.score == 0.0

    def test_to_dict_has_all_keys(self):
        r = RewardBreakdown()
        d = r.to_dict()
        for key in ["r1_task_completion", "r2_stakeholder_sentiment",
                    "r3_deadline_management", "r4_information_discipline",
                    "r5_strategic_coherence", "total"]:
            assert key in d