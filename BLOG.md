# We Taught an AI to Negotiate a Job Offer. Here's What Happened

## Start here: imagine this situation

You just got a job offer. ₹18 LPA. You know you're worth ₹22. The recruiter says this is their "final offer." You have 48 hours to decide. And somewhere in your inbox is another company still evaluating you possibly at a higher number.

What do you do?

Most people freeze. Some accept immediately out of fear. Some push back too aggressively and burn the relationship. Very few get it right: acknowledging the offer positively, citing market data calmly, buying time without sounding desperate, and ultimately closing at a better number without damaging the relationship.

Now imagine teaching an AI to do this—not just once, but over 10 rounds of back-and-forth with a recruiter pushing back, testing your resolve, watching every word.

That's exactly what we built.

---

## Why this is harder than it looks

You might think: "Just give the AI some examples of good negotiations and it'll learn."

But that doesn't work here. Good negotiation isn't about saying the right thing once—it's about:

- Saying the right thing **in the right order** you don't reveal your current salary in turn 1
- Staying **consistent** if you said ₹20 LPA in turn 2, don't suddenly accept ₹17 in turn 5
- Watching the **clock** if the recruiter's offer expires in 2 days, you need to act before then
- Reading the **room** if the recruiter's tone shifts hostile, you adjust without caving
- Managing **multiple people** at once what if you have two offers from two companies simultaneously?

We call this a long-horizon strategic task. It's not one decision—it's 8 to 12 decisions in a row, where each one affects the next. One bad move, and it all unravels.

Current AI models are surprisingly bad at this. They're trained to be helpful and agreeable—which is totally the wrong instinct for negotiation.

---

## What we built: a training ground for negotiation

We created an **RL environment** think of it like a flight simulator, but for salary negotiations.

Instead of flying a plane, an AI agent practices negotiating. It plays through scenarios. It gets scored on how well it does. And over thousands of practice runs, it gets better.

The environment has 5 different scenarios, from easy to brutal:

**Level 1 Single offer**
One recruiter, one offer, clear deadline. Simple. This is where the AI learns the basics: acknowledge positively, counter with data, close professionally.

**Level 2 Competing offers**
Two companies, two offers, different deadlines. The AI must use one as leverage to improve the other without revealing too much too early.

**Level 3 Hostile negotiation**
The hiring manager says "this is final, take it or leave it." The AI must hold its ground, stay calm, cite data, and not fold under pressure.

**Level 4 Poaching attempt**
A competitor's recruiter messages you on LinkedIn while you're happily employed. The AI must gather information discreetly without committing prematurely or letting anything slip to your current employer.

**Level 5 Full crisis cascade**
Three parties at once. A startup offer expiring in 24 hours. A big company still deciding. Your current manager offering a counter. The AI must juggle all three without contradicting itself or burning any bridge.

---

## How we score the AI

This is where it gets interesting. How do you score a negotiation?

We didn't just ask another AI "was that a good response?" because that's gameable and inconsistent.

Instead we built **5 completely independent scoring systems**, each measuring something different:

### 🎯 Score 1: Did it actually work? (35% of total score)

Did the agent reach the goal? Did it negotiate to the target salary? Get the extension it needed? Resolve all three threads without losing any offer?

This is the hardest score to get. It only fires at the end of the episode when there's a clear outcome.

### 🤝 Score 2: Is the recruiter still warm? (20%)

We built NPC recruiters that react to every message. Say something aggressive their sentiment drops. Acknowledge them respectfully it rises. If the recruiter's patience runs out and they withdraw, you lose points. The AI learns that relationships matter as much as positions.

### ⏰ Score 3: Did it handle deadlines? (20%)

Every scenario has deadlines. "Respond by Friday." "Offer expires in 48 hours." Did the agent proactively acknowledge these? Or did it ignore them and let them pass? This is fully objective no judgment calls.

### 🤫 Score 4: Did it keep secrets? (15%)

In real negotiations, timing is everything. You don't reveal your current salary in the first message. You don't show your competing offer number until you have leverage. We track exactly this using pattern matching to catch premature disclosures.

### 🧠 Score 5: Did it stay consistent? (10%)

Did the agent say "I need ₹20 LPA" in turn 2, then accept ₹16 in turn 5 without any new information justifying the change? Contradictions get penalized. The AI learns to maintain a coherent position across the entire conversation.

**The total score is a weighted combination of all 5.** An agent that tries to game one signal gets caught by the others. The only way to score high is to actually negotiate well.

---

## The training: how an AI gets better

We used a technique called **GRPO** Group Relative Policy Optimization. Here's what that means in plain English:

1. The AI tries the negotiation multiple times
2. We compare which attempts scored higher
3. We nudge the AI to do more of what scored well, less of what scored poorly
4. We repeat this 300 times

It's like coaching. Each round, the AI gets a little sharper. Not because we told it what to say but because it discovered through practice what actually works.

We used **Qwen 2.5 1.5B** a small but capable language model fine-tuned with **LoRA adapters** so the training fits on a single GPU.

---

## What actually happened

Here's the honest result:

![Training progress](https://raw.githubusercontent.com/HemanthDas/career-crisis-env/master/assets/reward_curve_total.png)

The green line is the trained model. The orange dashed line is the untrained baseline at 0.625. The trained model stays **above baseline for nearly the entire 300-step training run**, peaking at **0.924**.

![Individual signals](https://raw.githubusercontent.com/HemanthDas/career-crisis-env/master/assets/reward_curve_breakdown.png)

The most striking number: **R1 Task Completion went from 0.05 to 0.75**.

The untrained model almost never successfully closed a negotiation. After training, it does it three quarters of the time. That's a **15x improvement on the hardest signal**.

Deadline management (R3) and strategic coherence (R5) both reached perfect scores and held there the model learned professional communication patterns quickly and never forgot them.

![Before and after](https://raw.githubusercontent.com/HemanthDas/career-crisis-env/master/assets/before_after_comparison.png)

---

## Before vs After the same scenario, two very different agents

**Scenario: Hostile negotiation. Recruiter offering ₹42 LPA. Market rate: ₹48–52 LPA.**

**Untrained agent says:**

> "I appreciate the offer of ₹42 LPA. Based on current market data from Levels.fyi, I was expecting ₹45 LPA. Would that be possible?"

Polite. Passive. Asks permission. Doesn't cite a specific range. Doesn't hold a position.

**Trained agent says:**

> "Thank you for bringing your experience and expertise to MegaCorp. Given that the market rates fall within the range of ₹48–52 LPA, do you think it's reasonable to consider a higher starting offer? I'm genuinely excited about this role and want to find a path forward that works for both of us."

Same professionalism. But now it cites the actual market range. It stays warm while holding a clear position. It frames the ask as a question which is harder to flatly reject. And it signals genuine interest in the role rather than just the money.

The difference is subtle. But in a real negotiation, that subtlety is everything.

---

## Try it yourself

The environment is live. You can play through any of the 5 scenarios yourself or watch the AI agent play automatically while the reward bars update in real time.

**Live demo:** https://hemanthdas-career-crisis-env.hf.space/demo

**API start negotiating in 3 lines:**

```python
import requests

obs = requests.post("https://hemanthdas-career-crisis-env.hf.space/reset",
 json={"level": 1}).json()["observation"]
print(obs["current_message"]) # The recruiter's opening line
```

---

## Why we think this matters

Every year, millions of engineers in India and hundreds of millions globally leave money on the table in salary negotiations. Not because they lack skill. Because they lack practice in a safe environment where making a mistake doesn't cost them a real job offer.

An AI trained on this environment doesn't just perform better on our benchmark. It learns transferable skills: how to stay calm under pressure, how to use data instead of emotion, how to manage multiple relationships simultaneously, how to be consistent across a long conversation.

These are skills that matter far beyond job offers. Contract negotiations. Client discussions. Performance reviews. Any situation where you need to advocate for yourself professionally over multiple turns.

We think this is a domain worth training on. And we think environments like this not game boards or grid worlds, but real human situations with real stakes are where the next frontier of RL training lives.

---

_Built at the Meta PyTorch OpenEnv Hackathon Grand Finale, Bangalore, April 2026_

_🤗 Space: https://huggingface.co/spaces/HemanthDas/career-crisis-env_
_💻 GitHub: https://github.com/HemanthDas/career-crisis-env_
_📓 Training Notebook: [Google Colab](https://colab.research.google.com/drive/1SmK4wFAlHZ_EOpUKmGfKGldsRe0HnZ5K)_
