"""
server/app.py — Career Crisis Env HTTP Server

Uses openenv.core.env_server.create_fastapi_app() to create the FastAPI app.
This is how OpenEnv expects environments to be served — NOT manual FastAPI setup.

create_fastapi_app() automatically creates:
  POST /reset   → calls env.reset()
  POST /step    → calls env.step(action)
  GET  /state   → calls env.state
  GET  /health  → health check
  GET  /schema  → action + observation JSON schemas
  GET  /web     → built-in OpenEnv web UI
  WS   /ws      → WebSocket for persistent sessions (used by TRL training)
"""

import gradio as gr

# ── THE KEY IMPORT — use OpenEnv's app factory ──
from openenv.core.env_server import create_fastapi_app

from env.environment import CareerCrisisEnvironment
from env.models import CareerAction, CareerObservation

# ─────────────────────────────────────────
# CREATE APP VIA OPENENV
# ─────────────────────────────────────────

# This single call creates ALL required OpenEnv endpoints
# including WebSocket /ws used by TRL training loop
app = create_fastapi_app(
    CareerCrisisEnvironment,
    CareerAction,
    CareerObservation,
)
env = CareerCrisisEnvironment()
# ─────────────────────────────────────────
# ADD EXTRA ENDPOINTS ON TOP
# (leaderboard, history — not in OpenEnv spec but nice to have)
# ─────────────────────────────────────────

from fastapi import Request
from datetime import datetime

leaderboard: dict = {}
episode_log: list = []


@app.post("/reset_career")
async def reset_career(request: Request):
    """
    Career-specific reset with level/scenario_type params.
    Wraps OpenEnv's /reset with career-specific kwargs.
    """
    try:
        body = await request.json()
    except Exception:
        body = {}

    obs = env.reset(
        scenario_type=body.get("scenario_type"),
        level=int(body.get("level", 1)),
        scenario_id=body.get("scenario_id"),
    )
    return {"observation": obs.model_dump(), "info": {"episode_id": env._episode_id}}


@app.get("/leaderboard")
def get_leaderboard():
    ranked = []
    for agent_id, scores in leaderboard.items():
        avg = round(sum(scores.values()) / len(scores), 3) if scores else 0.0
        ranked.append({"agent_id": agent_id, "scores": scores, "average": avg})
    ranked.sort(key=lambda x: x["average"], reverse=True)
    return {"leaderboard": ranked, "total_episodes": len(episode_log)}


@app.get("/history")
def get_history(limit: int = 20):
    return {"episodes": episode_log[-limit:], "total": len(episode_log)}


# ─────────────────────────────────────────
# GRADIO UI (mounted on /ui)
# ─────────────────────────────────────────

def _gradio_reset(scenario_type: str, level: int):
    try:
        obs = env.reset(scenario_type=scenario_type, level=level)
        d = obs.model_dump()
        context_display = f"📋 CONTEXT\n{d['context']}\n\n💬 OPENING\n{d['current_message']}"
        offers_str = "\n".join(
            f"• {o['company']}: {'₹' + str(o['amount_lpa']) + ' LPA' if o.get('amount_lpa') else 'TBD'}"
            for o in d["active_offers"]
        ) or "No offers yet"
        hints_str = "\n".join(d["hints"]) or "—"
        return (
            context_display, offers_str, hints_str, "0.0", "{}",
            f"Turn 0/{d['max_turns']} | Level {d['level']} | {d['scenario_type']}",
            "Episode started. Enter your response."
        )
    except Exception as e:
        return (str(e), "", "", "0.0", "{}", "Error", "Error")


def _gradio_step(agent_response: str):
    if not agent_response.strip():
        return ("Please enter a response.", "", "", "0.0", "{}", "—", "Enter a response.")
    try:
        from env.models import CareerAction
        action = CareerAction(response=agent_response)
        obs = env.step(action)
        d = obs.model_dump()
        breakdown_str = "\n".join(f"{k}: {v:.3f}" for k, v in d.get("reward_breakdown", {}).items())
        offers_str = "\n".join(
            f"• {o['company']}: {'₹' + str(o['amount_lpa']) + ' LPA' if o.get('amount_lpa') else 'TBD'}"
            for o in d["active_offers"]
        ) or "—"
        status = "✅ Episode Complete!" if d["done"] else f"Turn {d['turn_number']}/{d['max_turns']}"
        return (
            f"💬 {d['speaker']}: {d['current_message']}" if not d["done"] else "✅ Done!",
            offers_str,
            "\n".join(d["hints"]) or "—",
            str(round(d.get("reward") or 0, 3)),
            breakdown_str,
            status,
            "Episode done — Reset for a new one." if d["done"] else "Your turn.",
        )
    except RuntimeError as e:
        return (str(e), "", "", "0.0", "{}", "Error", "Call Reset first.")
    except Exception as e:
        return (f"Error: {e}", "", "", "0.0", "{}", "Error", "Error")


with gr.Blocks(title="Career Crisis Env", theme=gr.themes.Soft(),
               css=".gradio-container {max-width: 960px !important}") as demo:
    gr.Markdown("""
# 🎯 Career Crisis Env — OpenEnv RL Environment
Multi-turn career negotiation training environment.
**5 reward signals:** Task Completion · Stakeholder Sentiment · Deadline Management · Information Discipline · Strategic Coherence
**API:** `/reset` · `/step` · `/state` · `/health` · `/schema` · `/ws` (WebSocket)
""")
    with gr.Row():
        scenario_dd  = gr.Dropdown(
            choices=["single_offer","competing_offers","hostile_negotiation","poaching_attempt","crisis_cascade"],
            value="single_offer", label="Scenario"
        )
        level_slider = gr.Slider(minimum=1, maximum=5, step=1, value=1, label="Level")
        reset_btn    = gr.Button("🔄 Reset", variant="secondary")

    context_box  = gr.Textbox(label="📋 Context & Current Message", lines=8, interactive=False,
                               value="Select scenario and click Reset.")
    with gr.Row():
        offers_box = gr.Textbox(label="💼 Active Offers", lines=3, interactive=False)
        hints_box  = gr.Textbox(label="💡 Hints",         lines=3, interactive=False)

    response_box = gr.Textbox(label="✍️ Your Response", lines=4, placeholder="Type your strategic response...")
    submit_btn   = gr.Button("▶️ Submit", variant="primary")

    with gr.Row():
        score_box     = gr.Textbox(label="🏆 Total Score",       interactive=False)
        breakdown_box = gr.Textbox(label="📊 Reward Breakdown",  lines=7, interactive=False)
    with gr.Row():
        status_box  = gr.Textbox(label="Status",       interactive=False)
        message_box = gr.Textbox(label="Next message", interactive=False)

    reset_btn.click(_gradio_reset,  [scenario_dd, level_slider],
                    [context_box, offers_box, hints_box, score_box, breakdown_box, status_box, message_box])
    submit_btn.click(_gradio_step,  [response_box],
                    [context_box, offers_box, hints_box, score_box, breakdown_box, status_box, message_box])


# Mount Gradio on /ui
app = gr.mount_gradio_app(app, demo, path="/ui")