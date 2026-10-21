"""In-process documents, graph walk, and bag-of-words retrieval.

Neo4j and pgvector are not connected. Documents are filtered by tenant before scoring.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from knowledge.paths import dataset_dir

_TOKEN = re.compile(r"[a-z0-9][a-z0-9\-]*")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def load_documents(path: Path | None = None) -> list[dict]:
    source = path or (dataset_dir() / "documents.json")
    document = json.loads(source.read_text(encoding="utf-8"))
    return list(document["documents"])


def load_edges(path: Path | None = None) -> list[dict]:
    source = path or (dataset_dir() / "graph.json")
    document = json.loads(source.read_text(encoding="utf-8"))
    return list(document["edges"])


def load_cases(path: Path | None = None) -> list[dict]:
    source = path or (dataset_dir() / "cases.json")
    document = json.loads(source.read_text(encoding="utf-8"))
    return list(document["cases"])


def retrieve(tenant: str, query: str, documents: list[dict] | None = None, limit: int = 3) -> list[dict]:
    corpus = documents if documents is not None else load_documents()
    query_tokens = set(_tokens(query))
    ranked = []
    for document in corpus:
        if document["tenant"] != tenant:
            continue
        overlap = len(query_tokens.intersection(_tokens(document["text"] + " " + document["title"])))
        ranked.append((overlap, document["id"], document))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return [item[2] for item in ranked[:limit] if item[0] > 0]


def walk(tenant: str, start: str, edges: list[dict] | None = None, hops: int = 3) -> list[str]:
    graph = edges if edges is not None else load_edges()
    seen: list[str] = []
    frontier = [start]
    for _ in range(hops):
        nxt: list[str] = []
        for node in frontier:
            for edge in graph:
                if edge["tenant"] != tenant or edge["source"] != node:
                    continue
                row = f"{edge['source']} {edge['relation']} {edge['target']}"
                if row not in seen:
                    seen.append(row)
                nxt.append(edge["target"])
        frontier = nxt
    return seen
