import json
import os
import time
import uuid
from typing import Any

import chromadb
from chromadb.config import Settings

from src.config import Config


class SemanticCache:
    def __init__(self, db_path: str | None = None, collection_name: str = "analysis_cache"):
        self.db_path = db_path or os.path.join(Config().workdir, "vector_db")
        self.collection_name = collection_name

        chroma_host = os.getenv("CHROMA_HOST", "").strip()
        chroma_port = int(os.getenv("CHROMA_PORT", "8000"))

        if chroma_host:
            self.client = chromadb.HttpClient(
                host=chroma_host,
                port=chroma_port,
                settings=Settings(anonymized_telemetry=False),
            )
        else:
            os.makedirs(self.db_path, exist_ok=True)
            self.client = chromadb.PersistentClient(
                path=self.db_path,
                settings=Settings(anonymized_telemetry=False),
            )

        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def recall(self, query_vector: list[float], ticker: str | None = None, limit: int = 4) -> list[dict[str, Any]]:
        where = {"ticker": ticker} if ticker else None
        result = self.collection.query(
            query_embeddings=[query_vector],
            n_results=limit,
            where=where,
        )

        metadatas = result.get("metadatas", [[]])[0]
        documents = result.get("documents", [[]])[0]
        distances = result.get("distances", [[]])[0]
        ids = result.get("ids", [[]])[0]

        rows: list[dict[str, Any]] = []
        for idx, metadata in enumerate(metadatas):
            row = dict(metadata or {})
            row["id"] = ids[idx] if idx < len(ids) else ""
            row["document"] = documents[idx] if idx < len(documents) else ""
            if idx < len(distances):
                row["distance"] = distances[idx]
                row["score"] = 1.0 - float(distances[idx])
            rows.append(row)
        return rows

    def save_episode(
        self,
        ticker: str,
        summary: str,
        embedding: list[float],
        recommendation: str,
        confidence: str,
        last_price: float,
        prediction: dict[str, Any],
        news_sentiment: str = "",
        final_report: str = "",
        thread_id: str | None = None,
        use_fmi: bool = False,
        created_at_ts: int | None = None,
        llm_model: str = "",
        embed_model: str = "",
        news_empty: bool = False,
    ) -> None:
        self.collection.add(
            ids=[str(uuid.uuid4())],
            embeddings=[embedding],
            documents=[final_report or summary],
            metadatas=[
                {
                    "ticker": ticker,
                    "summary": summary,
                    "final_report": final_report or summary,
                    "recommendation": recommendation,
                    "confidence": confidence,
                    "last_price": float(last_price),
                    "prediction_json": json.dumps(prediction),
                    "news_sentiment": news_sentiment,
                    "thread_id": thread_id or "",
                    "use_fmi": bool(use_fmi),
                    "llm_model": llm_model,
                    "embed_model": embed_model,
                    "news_empty": bool(news_empty),
                    "created_at_ts": int(created_at_ts or time.time()),
                }
            ],
        )
