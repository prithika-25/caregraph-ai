from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
import uuid

from graph_rag import GraphRAGEngine
from escalation_model import EscalationPredictor

app = FastAPI(title="CareGraph AI", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = GraphRAGEngine()
predictor = EscalationPredictor()

# in-memory session store for the demo (would be Redis/Postgres in production)
SESSIONS = {}


class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    sources: list
    escalation: dict
    suggested_action: str
    turn_count: int


def build_reply(query: str, sources: list, risk_level: str) -> str:
    if not sources:
        opener = ("I couldn't find a confident match in the knowledge base for that yet — "
                   "I'm routing this to a support specialist so you're not stuck waiting on me.")
        return opener

    lead = sources[0]
    opener_by_risk = {
        "high": "I can see this hasn't been resolved and that's genuinely frustrating — here's what I can confirm right now, and I'm flagging this for priority human review in parallel: ",
        "medium": "Thanks for the details — here's what I found, and I'll keep an eye on this thread: ",
        "low": "Happy to help — here's what I found: ",
    }
    reply = opener_by_risk[risk_level] + lead["content"]

    # if a graph-expanded (multi-hop) related article was pulled in, surface it —
    # this is the compound-issue handling a flat retriever would miss
    related = [s for s in sources[1:] if s["hop"] == "graph-expanded"]
    if related:
        reply += f" Related to this, on {related[0]['title'].lower()}: {related[0]['content']}"

    return reply


def suggested_action(risk_level: str, sources: list) -> str:
    if risk_level == "high":
        return "auto_escalate_to_tier2"
    if risk_level == "medium" and sources and sources[0]["category"] in {"billing", "security"}:
        return "flag_for_review"
    if not sources:
        return "auto_escalate_to_tier2"
    return "auto_resolved"


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/graph")
def graph(query: str):
    """Returns the retrieval graph trace for a query, used by the frontend's live graph view."""
    return engine.explain_graph(query)


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    session = SESSIONS.setdefault(session_id, {"turns": [], "count": 0})
    session["count"] += 1
    session["turns"].append(req.message)

    sources = engine.retrieve(req.message)

    # escalation is scored on the full conversation-so-far, not just the last
    # message, so repeated contact ("again", "still") compounds risk correctly
    conversation_text = " ".join(session["turns"])
    escalation = predictor.predict(conversation_text)

    reply = build_reply(req.message, sources, escalation["risk_level"])
    action = suggested_action(escalation["risk_level"], sources)

    return ChatResponse(
        session_id=session_id,
        reply=reply,
        sources=sources,
        escalation=escalation,
        suggested_action=action,
        turn_count=session["count"],
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
