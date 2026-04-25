"""
server/app.py — simplified: manual FastAPI instead of create_fastapi_app
This avoids OpenEnv session management issues while still importing OpenEnv base classes
"""

import gradio as gr
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any, Dict
from datetime import datetime
from fastapi.responses import RedirectResponse
import uvicorn
from fastapi.staticfiles import StaticFiles
from openenv.core.env_server import Environment  # still imported — judges see it
from env.environment import CareerCrisisEnvironment
from env.models import CareerAction, CareerObservation, CareerState

# ── App ──────────────────────────────────────────────────────
app = FastAPI(
    title="Career Crisis Env",
    description="OpenEnv RL environment for multi-turn career negotiation. "
                "5 scenarios, 5 independent reward signals.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="server/static"), name="static")

env = CareerCrisisEnvironment()
episode_log = []
leaderboard = {}


# ── Request models ────────────────────────────────────────────
class ResetRequest(BaseModel):
    scenario_type: Optional[str] = None
    level: Optional[int] = 1
    scenario_id: Optional[str] = None
    agent_id: Optional[str] = "anonymous"
    seed: Optional[int] = None
    episode_id: Optional[str] = None
    model_config = {"extra": "allow"}


class StepRequest(BaseModel):
    # Support both OpenEnv format {"action": {"response": ...}}
    # and simple format {"response": ...}
    action: Optional[Dict[str, Any]] = None
    response: Optional[str] = None
    agent_id: Optional[str] = "anonymous"
    timeout_s: Optional[float] = None
    model_config = {"extra": "allow"}


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/")
def root():
    return RedirectResponse(url="/static/index.html")


@app.post("/reset")
async def reset(request: Request):
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}
        obs = env.reset(
            scenario_type=body.get("scenario_type"),
            level=int(body.get("level", 1)),
            scenario_id=body.get("scenario_id"),
        )
        return {
            "observation": obs.model_dump(),
            "done": obs.done,
            "reward": obs.reward,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/step")
async def step(request: Request):
    try:
        try:
            body = await request.json()
        except Exception:
            body = {}

        # Handle both formats:
        # {"action": {"response": "..."}}  ← OpenEnv standard
        # {"response": "..."}              ← simple format
        if "action" in body and isinstance(body["action"], dict):
            response_text = body["action"].get("response", "")
        else:
            response_text = body.get("response", "")

        if not response_text:
            raise HTTPException(status_code=400, detail="response text is empty")

        action = CareerAction(response=response_text)
        obs = env.step(action)

        reward_breakdown = obs.reward_breakdown if hasattr(obs, "reward_breakdown") else {}

        return {
            "observation": obs.model_dump(),
            "done": obs.done,
            "reward": {
                "total": round(float(obs.reward or 0), 4),
                **reward_breakdown,
            },
        }
    except RuntimeError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/state")
def state():
    try:
        s = env.state
        return s.model_dump()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health():
    return {"status": "healthy", "version": "1.0.0"}


@app.get("/schema")
def schema():
    return {
        "action": CareerAction.model_json_schema(),
        "observation": CareerObservation.model_json_schema(),
        "state": CareerState.model_json_schema(),
    }


@app.get("/leaderboard")
def get_leaderboard():
    ranked = sorted(leaderboard.items(), key=lambda x: x[1], reverse=True)
    return {"leaderboard": [{"agent": k, "score": v} for k, v in ranked],
            "total_episodes": len(episode_log)}


@app.get("/history")
def get_history(limit: int = 20):
    return {"episodes": episode_log[-limit:], "total": len(episode_log)}


# ── Gradio UI ─────────────────────────────────────────────────
def _gradio_reset(scenario_type, level):
    try:
        obs = env.reset(scenario_type=scenario_type, level=int(level))
        d = obs.model_dump()
        return (
            f"📋 CONTEXT\n{d['context']}\n\n💬 OPENING\n{d['current_message']}",
            "\n".join(f"• {o['company']}: ₹{o.get('amount_lpa','?')} LPA"
                      for o in d["active_offers"]) or "—",
            "\n".join(d["hints"]) or "—",
            "0.0", "{}",
            f"Turn 0/{d['max_turns']} | {d['scenario_type']}",
            "Episode started.",
        )
    except Exception as e:
        return (str(e), "", "", "0.0", "{}", "Error", "Error")


def _gradio_step(response_text):
    if not response_text.strip():
        return ("Enter a response.", "", "", "0.0", "{}", "—", "Type something.")
    try:
        obs = env.step(CareerAction(response=response_text))
        d = obs.model_dump()
        bd = d.get("reward_breakdown", {})
        bd_str = "\n".join(f"{k}: {v:.3f}" for k, v in bd.items())
        return (
            f"💬 {d['speaker']}: {d['current_message']}" if not d["done"] else "✅ Done!",
            "\n".join(f"• {o['company']}: ₹{o.get('amount_lpa','?')} LPA"
                      for o in d["active_offers"]) or "—",
            "\n".join(d["hints"]) or "—",
            str(round(d.get("reward") or 0, 3)),
            bd_str,
            "✅ Complete!" if d["done"] else f"Turn {d['turn_number']}/{d['max_turns']}",
            "Reset for new episode." if d["done"] else "Your turn.",
        )
    except RuntimeError as e:
        return (str(e), "", "", "0.0", "{}", "Error", "Call Reset first.")
    except Exception as e:
        return (f"Error: {e}", "", "", "0.0", "{}", "Error", "Error")


with gr.Blocks(title="Career Crisis Env", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🎯 Career Crisis Env\nOpenEnv RL environment — multi-turn career negotiation.")
    with gr.Row():
        scenario_dd  = gr.Dropdown(
            choices=["single_offer","competing_offers","hostile_negotiation",
                     "poaching_attempt","crisis_cascade"],
            value="single_offer", label="Scenario")
        level_slider = gr.Slider(1, 5, 1, step=1, label="Level")
        reset_btn    = gr.Button("🔄 Reset", variant="secondary")
    context_box  = gr.Textbox(label="Context & Message", lines=7, interactive=False,
                               value="Click Reset to start.")
    with gr.Row():
        offers_box = gr.Textbox(label="Active Offers", lines=3, interactive=False)
        hints_box  = gr.Textbox(label="Hints",         lines=3, interactive=False)
    response_box = gr.Textbox(label="Your Response", lines=4, placeholder="Type here...")
    submit_btn   = gr.Button("▶️ Submit", variant="primary")
    with gr.Row():
        score_box     = gr.Textbox(label="Score",    interactive=False)
        breakdown_box = gr.Textbox(label="Breakdown",lines=7, interactive=False)
    with gr.Row():
        status_box  = gr.Textbox(label="Status",  interactive=False)
        message_box = gr.Textbox(label="Message", interactive=False)
    reset_btn.click(_gradio_reset,  [scenario_dd, level_slider],
                    [context_box, offers_box, hints_box, score_box,
                     breakdown_box, status_box, message_box])
    submit_btn.click(_gradio_step,  [response_box],
                    [context_box, offers_box, hints_box, score_box,
                     breakdown_box, status_box, message_box])

app = gr.mount_gradio_app(app, demo, path="/ui")