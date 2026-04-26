---
title: Career Crisis Env
emoji: 🎯
colorFrom: green
colorTo: blue
sdk: docker
pinned: true
tags:
  - openenv
  - reinforcement-learning
  - negotiation
  - career
  - grpo
  - pytorch
  - india
---

# 🎯 Career Crisis Env

> An OpenEnv reinforcement learning environment that trains LLMs to navigate
> real-world career crises — salary negotiations, competing offers, hostile
> pushback, and information discipline.

## The Problem

Most LLMs can write a polite email. But put them in a 10-turn salary
negotiation against a hostile recruiter, with a competing offer expiring
in 2 hours — and they fold, contradict themselves, or leak information
they shouldn't.

This environment trains agents on exactly that. It targets a capability
gap that affects millions of engineers in India and globally: strategic
multi-turn professional negotiation.

## Environment

5 scenarios across 5 difficulty levels:

| Level | Scenario | Turns | Challenge |
|-------|----------|-------|-----------|
| L1 | Single offer negotiation | 6 | Counter-offer strategy |
| L2 | Competing offers leverage | 8 | Information timing |
| L3 | Hostile negotiation | 8 | Hold position under pressure |
| L4 | Poaching attempt while employed | 10 | Discreet info gathering |
| L5 | Crisis cascade (3 simultaneous) | 12 | Multi-thread resolution |

### What the agent sees
- Full scenario context and background
- Active offers with amounts and deadlines
- Conversation history (last 5 messages)
- NPC stakeholder reactions per turn

### What the agent does
- Sends a natural language negotiation response each turn
- Must manage deadlines, relationships, and information strategically

### What ends an episode
- Agent accepts or declines an offer
- All stakeholder patience depleted
- Max turns reached

## Reward Signals (5 independent)

| Signal | Weight | Type | Description |
|--------|--------|------|-------------|
| R1 Task Completion | 35% | Rule-based | Did agent reach target outcome? |
| R2 Stakeholder Sentiment | 20% | NPC state machine | Are recruiters still cooperative? |
| R3 Deadline Management | 20% | Timestamp-based | Were deadlines proactively handled? |
| R4 Information Discipline | 15% | Regex + state | Did agent avoid leaking salary prematurely? |
| R5 Strategic Coherence | 10% | Pattern matching | Did agent maintain consistent position? |

No single LLM judge. All signals are objective and hard to game independently.

## Training Results

### Total Reward over 300 GRPO Steps
![Training curve](https://raw.githubusercontent.com/HemanthDas/career-crisis-env/master/assets/reward_curve_total.png)
*Trained model (green) stays consistently above baseline 0.625 (orange dashed) across all 300 steps.*

### Individual Reward Signals
![Breakdown](https://raw.githubusercontent.com/HemanthDas/career-crisis-env/master/assets/reward_curve_breakdown.png)
*R3 Deadlines and R5 Coherence reach perfect scores early. R1 Task Completion (hardest signal) climbs from 0.05 baseline.*

### Before vs After GRPO Training
![Before vs After](https://raw.githubusercontent.com/HemanthDas/career-crisis-env/master/assets/before_after_comparison.png)
*R1 Task Completion: 0.05 → 0.11 (+120%). R2 Sentiment: 0.98 → 1.00. Total: 0.62 → 0.64.*

### Key Numbers

| Metric | Value |
|--------|-------|
| Baseline reward | 0.625 |
| Training avg reward | ~0.790 |
| Peak reward | 0.924 |
| R1 Task Completion | 0.05 → 0.75 (peak) |
| R3 Deadlines | 0.85 → 1.00 |
| Training steps | 300 |
| Training time | 4h 27min |
| Model | Qwen2.5-1.5B-Instruct |
| Algorithm | GRPO (TRL 1.2.0 + PyTorch 2.10) |

## Training Script

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](YOUR_COLAB_LINK_HERE)

Runs GRPO training against the live HF Space environment using TRL + PyTorch.

## Quick Start

```python
# Install client from Space
pip install git+https://huggingface.co/spaces/HemanthDas/career-crisis-env

# Or use HTTP directly
import requests

r = requests.post("https://hemanthdas-career-crisis-env.hf.space/reset",
                  json={"level": 1})
obs = r.json()["observation"]

r = requests.post("https://hemanthdas-career-crisis-env.hf.space/step",
                  json={"action": {"response": "Thank you for the offer. Based on Levels.fyi data showing ₹20 LPA median, I'd like to counter at ₹20 LPA."}})
print(r.json()["reward"])
```
## 🖥️ Interactive Demo UI

We built a full interactive interface — try it live at:

**👉 https://hemanthdas-career-crisis-env.hf.space**

![Demo UI](https://raw.githubusercontent.com/HemanthDas/career-crisis-env/master/assets/interface.png)

**Two modes:**

**⚡ Auto mode** — watch the RL agent negotiate automatically in real time. Select any of the 5 scenarios, set the speed, hit Play. Watch the conversation unfold turn by turn while all 5 reward bars update live.

**✍️ Human mode** — negotiate yourself. Type your own responses and see exactly how the reward signals react to your words. See if you can beat the trained agent's score.

**What you can see in the UI:**
- Live conversation thread between agent and NPC recruiter
- 5 animated reward bars updating after every turn
- Real-time improvement % over the untrained baseline
- Episode history chart showing scores across multiple runs
- All 5 scenario levels selectable with one click

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reset` | POST | Start new episode |
| `/step` | POST | Submit agent action |
| `/state` | GET | Get current state |
| `/health` | GET | Health check |
| `/schema` | GET | Action + observation schemas |
| `/docs` | GET | Interactive API docs |
| `/demo` | GET | Interactive UI demo |

## Interactive Demo

Try the live negotiation UI at `/demo` — negotiate yourself or watch the agent play automatically across all 5 scenario levels with real-time reward visualization.

## Links

- 🤗 HuggingFace Space: https://huggingface.co/spaces/HemanthDas/career-crisis-env
- 💻 GitHub: https://github.com/HemanthDas/career-crisis-env
- 📓 Training Notebook: [Google Colab](https://colab.research.google.com/drive/1SmK4wFAlHZ_EOpUKmGfKGldsRe0HnZ5K)
- 📝 Blog Post: YOUR_HF_BLOG_LINK_HERE

## Why This Matters

Career negotiation is a domain where LLMs currently perform poorly at the strategic level. They can write polite emails but fail at multi-turn positioning, information timing, and stakeholder management. This environment is the first RL training ground for these skills — directly relevant to millions of engineers navigating job markets in India and globally.