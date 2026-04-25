

from __future__ import annotations
from typing import Any

# ── THE KEY IMPORT — use OpenEnv's HTTP client base ──
from openenv_core import HTTPEnvClient, StepResult

from env.models import CareerAction, CareerObservation, CareerState


class CareerEnvClient(HTTPEnvClient[CareerAction, CareerObservation]):
    def _step_payload(self, action: CareerAction) -> dict:
        """Convert CareerAction to JSON payload for POST /step."""
        return {"action": {"response": action.response}}

    def _parse_result(self, payload: dict) -> StepResult[CareerObservation]:
        """Parse POST /step JSON response into StepResult[CareerObservation]."""
        obs_data = payload.get("observation", {})
        return StepResult(
            observation=CareerObservation(**obs_data),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> CareerState:
        return CareerState(**payload)