# CareGraph AI

A customer care bot for the **FlowZint AI Hackathon 2026** (Customer Care Bot track).

## Why I built it this way

Most customer care bots do one thing: match your message to an FAQ answer. That's fine
until someone writes something like *"my payment failed again and I still haven't
gotten my refund, fix this now or I'm cancelling."* That's actually two problems
(payment failure + refund) wrapped in a message that's clearly about to escalate — and
a plain FAQ-matcher usually only catches the first part and has no idea the second part
matters.

So I built two things instead of one:

1. A **graph-based retrieval system** — the knowledge base isn't a flat list, it's a
   graph where articles are linked by shared topics/tags. When a query comes in, it
   first finds the closest match (TF-IDF), then walks the graph to pull in anything
   *connected* to that match. So compound issues get a complete answer instead of half
   of one.
2. An **escalation risk model** — a logistic regression classifier that scores every
   message on things like negative sentiment, repeat-contact language ("again",
   "still"), threats ("cancel", "chargeback", "lawyer"), and shouting (caps ratio). It
   outputs a risk score and a suggested action: let the bot keep handling it, flag it
   for review, or escalate straight to a human.

The point isn't "the bot can answer questions" — every bot can do that. The point is it
knows when it *shouldn't* be the one answering anymore.

## How it's put together

```
backend/
  main.py              → FastAPI app, ties everything together (/chat, /graph, /health)
  graph_rag.py          → knowledge graph + retrieval logic
  escalation_model.py    → feature extraction + risk classifier
  data/kb.json            → the knowledge base (14 articles, tagged for the graph)
  models/                  → trained model gets saved here on first run
frontend/
  index.html               → chat UI, no build step, just open it in a browser
```

I kept the retrieval side on TF-IDF instead of embeddings on purpose — it runs fully
offline with no external API calls, which matters for a live demo where wifi is not
guaranteed to cooperate.

The escalation classifier is trained on a synthetic labeled dataset (calm / frustrated
/ angry message patterns, with realistic label noise so frustrated ≠ automatically
escalated). There's no public dataset for "will this support ticket escalate," and real
ticket logs are company-private, so this is the honest way to demonstrate the
approach — in production this function is the one place you'd swap in real historical
ticket outcomes.

## Running it

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

Then open `frontend/index.html` in a browser. That's it — no separate frontend build.
### Low-risk query
![Low risk query with graph retrieval] <img width="800" alt="image" src="https://github.com/user-attachments/assets/88a6456f-102f-4f38-9c16-b3c81b760e0a" />
### Compound issue — graph expansion in action
![Graph expansion for compound issue] <img width="800" alt="image" src="https://github.com/user-attachments/assets/aab5a639-bca1-4c80-93cd-267bbad6fbf6" />
### High-risk escalation
![High risk escalation with signal breakdown] <img width="800" alt="image" src="https://github.com/user-attachments/assets/370109a1-57c4-4cfc-9e8d-5756877f62fb" />

## What I'd add with more time

- Real sentence embeddings instead of TF-IDF once offline-friendliness stops mattering
- Persist sessions somewhere durable instead of an in-memory dict
- Feed actual resolved/escalated outcomes back into the classifier over time
- Escalation lexicon in more than one language
