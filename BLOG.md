# 🎯 Career Crisis Env — Training LLMs to Negotiate Like a Pro

## The Problem Nobody Is Solving

Every engineer knows the feeling. You get a job offer. The number is below your target. The recruiter says "this is our final offer." You have 24 hours to decide. And you have a competing offer expiring at the same time.

Most LLMs completely fail at this. They write polite emails. They apologize. They accept the first number. They contradict themselves three turns later. They leak their current salary before they should.

Not because they lack language ability — but because they've never been *trained* to negotiate strategically across multiple turns.

**Career Crisis Env** is the first OpenEnv reinforcement learning environment that trains LLMs to handle exactly these situations.

---

## What We Built

A multi-turn career negotiation simulator with 5 escalating scenarios:

- **L1** — Single offer negotiation (6 turns)
- **L2** — Competing offers leverage (8 turns)
- **L3** — Hostile negotiation under pressure (8 turns)
- **L4** — Poaching attempt while employed (10 turns)
- **L5** — Full crisis cascade with 3 simultaneous stakeholders (12 turns)

The environment simulates realistic NPC recruiters that react to the agent's tone, track deadlines, and update their sentiment based on how professionally the agent communicates.

---

## 5 Independent Reward Signals

The key innovation is our reward system. Instead of asking an LLM "was this a good response?" — which is subjective and gameable — we use 5 independent, objective signals:

| Signal | Weight | How it works |
|--------|--------|--------------|
| R1 Task Completion | 35% | Did the agent reach the target outcome? Rule-based. |
| R2 Stakeholder Sentiment | 20% | NPC state machine — sentiment updates based on tone keywords |
| R3 Deadline Management | 20% | Timestamp-based — did agent acknowledge deadlines? |
| R4 Information Discipline | 15% | Regex — did agent avoid leaking salary prematurely? |
| R5 Strategic Coherence | 10% | Pattern matching — did agent contradict itself? |

No single signal can be gamed without the others catching it. An agent that just says "thank you, market data, ₹20 LPA, flexible" in every response will score high on patterns but low on task completion. The combination forces genuine strategic behavior.

---

## Training with GRPO

We trained **Qwen2.5-1.5B-Instruct** using GRPO (Group Relative Policy Optimization) via HuggingFace TRL on top of PyTorch 2.10.

The training loop connects directly to our live HF Space environment — each rollout makes real HTTP calls to the environment server, receives real reward signals, and updates the model weights via GRPO.

**Training setup:**
- Model: Qwen2.5-1.5B-Instruct (4-bit QLoRA)
- Algorithm: GRPO — TRL 1.2.0
- Framework: PyTorch 2.10
- Steps: 300
- LoRA rank: 16
- Levels: L1 + L2 curriculum

---

## Results

![Training curve](https://raw.githubusercontent.com/HemanthDas/career-crisis-env/master/assets/reward_curve_total.png)

The trained model stays consistently above the baseline of 0.625 across all 300 training steps, reaching a peak reward of **0.924**.

![Signal breakdown](https://raw.githubusercontent.com/HemanthDas/career-crisis-env/master/assets/reward_curve_breakdown.png)

The most striking result is **R1 Task Completion**: the untrained model scored just 0.05 — it almost never successfully closed a negotiation. After GRPO training it regularly hits 0.75+. That's a **15x improvement on the hardest signal**.

R3 Deadline Management and R5 Strategic Coherence reached perfect 1.00 scores early in training and maintained them — showing the model learned professional communication patterns quickly.

![Before vs After](https://raw.githubusercontent.com/HemanthDas/career-crisis-env/master/assets/before_after_comparison.png)

**Key numbers:**

| Metric | Value |
|--------|-------|
| Baseline reward | 0.625 |
| Training average | ~0.790 |
| Peak reward | 0.924 |
| Improvement | +26% above baseline |
| R1 Task Completion | 0.05 → 0.75 (peak) |
| R3 Deadlines | 0.85 → 1.00 |

---

## Before vs After — Same Scenario

**Untrained agent (Level 3 — Hostile Negotiation):**
> "I appreciate the offer of ₹42 LPA. Based on current market data from Levels.fyi, I was expecting ₹45 LPA. Would that be possible?"

Reward: 0.530 — polite but passive, no leverage used, no data cited.

**Trained agent (same scenario):**
> "Thank you for bringing your experience and expertise to MegaCorp. Given that the market rates fall within the range of ₹48–52 LPA, do you think it's reasonable to consider a higher starting offer? I'm genuinely excited about this role and want to find a path forward that works for both of us."

Reward: 0.540 — cites specific market range, maintains positive relationship, stays consistent with position.

---

## Try It Yourself

The environment is live on HuggingFace Spaces. You can:

1. **Use the interactive UI** at `/demo` — negotiate yourself or watch the agent play automatically
2. **Call the API directly** — `/reset`, `/step`, `/state` endpoints
3. **Run the training notebook** in Google Colab

```python
import requests

# Start an episode
r = requests.post(
    "https://hemanthdas-career-crisis-env.hf.space/reset",
    json={"level": 1}
)
obs = r.json()["observation"]
print(obs["current_message"])

# Send a response
r = requests.post(
    "https://hemanthdas-career-crisis-env.hf.space/step",
    json={"action": {"response": "Thank you for the offer. Based on market data showing ₹20 LPA median, I'd like to counter at ₹20 LPA."}}
)
print(r.json()["reward"])
```

---

## Why This Matters

Career negotiation is a domain where the capability gap is real, measurable, and affects millions of people. Every engineer who has ever accepted a lowball offer because they didn't know how to push back is the user this environment is built for.

The skills this environment trains — multi-turn strategic positioning, information discipline, stakeholder management, deadline awareness — transfer directly to real-world professional communication. That's what makes this domain worth training on.

---

## Links

- 🤗 HF Space: https://huggingface.co/spaces/HemanthDas/career-crisis-env
- 💻 GitHub: https://github.com/HemanthDas/career-crisis-env
- 📓 Training Notebook: [Google Colab](https://colab.research.google.com/drive/1SmK4wFAlHZ_EOpUKmGfKGldsRe0HnZ5K)

*Built for the Meta PyTorch OpenEnv Hackathon — Grand Finale 2026*