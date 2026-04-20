import hashlib
import json
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.api.types import EmbeddingFunction, Embeddings


class LocalHashEmbedding(EmbeddingFunction):
    def __call__(self, input: Iterable[str]) -> Embeddings:
        return [self._embed(text) for text in input]

    def _embed(self, text: str) -> list[float]:
        vector = [0.0] * 256
        tokens = (text or "").lower().split()
        if not tokens:
            return vector

        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = digest[0]
            sign = 1.0 if digest[1] % 2 == 0 else -1.0
            vector[index] += sign

        scale = float(len(tokens))
        return [round(value / scale, 6) for value in vector]


class NewsroomMemory:
    def __init__(self, root: Path):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(self.root))
        self.embedding = LocalHashEmbedding()
        self.collection = self.client.get_or_create_collection(
            name="newsroom_memory",
            embedding_function=self.embedding,
        )

    def store_report(self, topic: str, report: str, sources: list[dict], overview: str):
        identifier = hashlib.md5(topic.encode("utf-8")).hexdigest()
        document = json.dumps(
            {
                "topic": topic,
                "overview": overview,
                "report": report[:1200],
                "sources": sources[:8],
            },
            ensure_ascii=True,
        )
        self.collection.upsert(
            ids=[identifier],
            documents=[document],
            metadatas=[{"topic": topic}],
        )

    def retrieve(self, topic: str, limit: int = 3) -> list[dict]:
        result = self.collection.query(query_texts=[topic], n_results=limit)
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]

        hits = []
        for doc, metadata in zip(documents, metadatas):
            try:
                payload = json.loads(doc)
            except json.JSONDecodeError:
                payload = {"report": doc, "overview": "", "sources": []}
            payload["memory_topic"] = (metadata or {}).get("topic", payload.get("topic", ""))
            hits.append(payload)
        return hits
