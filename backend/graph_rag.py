"""
CareGraph AI - Graph-RAG Retrieval Engine
------------------------------------------
Most "RAG chatbots" do flat top-k vector/keyword retrieval: one query, one
similarity pass, done. That breaks down for customer care, where a single
complaint often spans multiple linked issues (e.g. "payment failed AND I
want a refund AND now my account is locked").

This module builds a small knowledge graph over the KB where articles are
nodes and shared tags/entities are edges. Retrieval works in two hops:
  1. Seed hop: TF-IDF cosine similarity finds the most relevant article(s)
     for the raw query.
  2. Graph hop: we walk out from the seed node(s) along tag-shared edges to
     pull in *connected* articles that a flat retriever would miss, then
     re-rank the combined candidate pool by relevance to the query.

This lets the bot answer compound, multi-issue tickets with a single
coherent response instead of only addressing the most keyword-similar
sentence.
"""

import json
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

DATA_PATH = Path(__file__).parent / "data" / "kb.json"


class GraphRAGEngine:
    def __init__(self, kb_path: Path = DATA_PATH):
        with open(kb_path, "r") as f:
            self.kb = json.load(f)

        self.id_to_doc = {d["id"]: d for d in self.kb}
        self._build_graph()
        self._build_vectorizer()

    def _build_graph(self):
        """Nodes = KB articles. Edge weight = number of shared tags."""
        g = nx.Graph()
        for doc in self.kb:
            g.add_node(doc["id"], **doc)

        for i, a in enumerate(self.kb):
            for b in self.kb[i + 1:]:
                shared = set(a["tags"]) & set(b["tags"])
                if shared:
                    g.add_edge(a["id"], b["id"], weight=len(shared), shared=list(shared))
        self.graph = g

    def _build_vectorizer(self):
        corpus = [f"{d['title']} {d['content']} {' '.join(d['tags'])}" for d in self.kb]
        self.vectorizer = TfidfVectorizer(stop_words="english")
        self.doc_matrix = self.vectorizer.fit_transform(corpus)

    def _seed_retrieval(self, query: str, top_k: int = 2):
        q_vec = self.vectorizer.transform([query])
        sims = cosine_similarity(q_vec, self.doc_matrix).flatten()
        ranked = sims.argsort()[::-1][:top_k]
        return [(self.kb[i]["id"], float(sims[i])) for i in ranked if sims[i] > 0]

    def _graph_expand(self, seed_ids, max_hop_neighbors: int = 3):
        """Pull in neighbors of seed nodes, weighted by shared-tag strength."""
        candidates = {}
        for sid, score in seed_ids:
            candidates[sid] = max(candidates.get(sid, 0), score)
            for neighbor in self.graph.neighbors(sid):
                edge_w = self.graph[sid][neighbor]["weight"]
                # graph-hop score decays relative to the seed's own relevance
                hop_score = score * 0.5 * (edge_w / 3)
                candidates[neighbor] = max(candidates.get(neighbor, 0), hop_score)
        ranked = sorted(candidates.items(), key=lambda x: x[1], reverse=True)
        return ranked[: max_hop_neighbors + len(seed_ids)]

    def retrieve(self, query: str, top_k_seed: int = 2, max_results: int = 4):
        seeds = self._seed_retrieval(query, top_k=top_k_seed)
        if not seeds:
            return []
        expanded = self._graph_expand(seeds)
        results = []
        for doc_id, score in expanded[:max_results]:
            doc = self.id_to_doc[doc_id]
            results.append({
                "id": doc_id,
                "title": doc["title"],
                "content": doc["content"],
                "category": doc["category"],
                "relevance": round(score, 3),
                "hop": "seed" if doc_id in [s[0] for s in seeds] else "graph-expanded",
            })
        return results

    def explain_graph(self, query: str):
        """Returns the seed + expansion trace, used to render the live graph in the UI."""
        seeds = self._seed_retrieval(query, top_k=2)
        seed_ids = [s[0] for s in seeds]
        expanded = self._graph_expand(seeds)
        nodes = []
        edges = []
        included = [e[0] for e in expanded]
        for doc_id in included:
            nodes.append({
                "id": doc_id,
                "title": self.id_to_doc[doc_id]["title"],
                "type": "seed" if doc_id in seed_ids else "expanded",
            })
        for a in included:
            for b in included:
                if a < b and self.graph.has_edge(a, b):
                    edges.append({"source": a, "target": b, "shared": self.graph[a][b]["shared"]})
        return {"nodes": nodes, "edges": edges}


if __name__ == "__main__":
    engine = GraphRAGEngine()
    test_query = "my payment failed but money got deducted and now I want a refund"
    for r in engine.retrieve(test_query):
        print(r["hop"], "-", r["title"], "-", r["relevance"])
