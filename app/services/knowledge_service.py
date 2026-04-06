import hashlib
import re
import sys
import threading
from pathlib import Path

from ..config.settings import settings
from ..data.project_index import load_project_records

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
    "explain", "help", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "the", "to",
    "tell", "works", "working",
    "what", "when", "where", "who", "why", "with", "you", "your",
}


class KnowledgeService:
    """
    Vector-only knowledge retrieval for conversation mode.

    Local files are still loaded so they can seed the vector database, but answers
    are returned only from embedded records that exist in MongoDB.
    """

    def __init__(self):
        self.model = None
        self._np = None
        self.client = None
        self.db = None
        self.collection = None
        self.embedding_dim = None
        self.last_error = ""
        self.semantic_search_enabled = settings.enable_semantic_search
        self._cache_lock = threading.Lock()
        self._indexed_records: list[dict] = []
        self._embedding_matrix = None
        self._index_dirty = True
        self._curated_seed_checked = False
        curated_documents = self._load_local_documents()
        project_chunks = self._load_project_chunks()
        self.seed_version = "v2"
        self.curated_seed_documents = self._tag_source_kind(
            self._build_local_chunks(curated_documents),
            "curated",
        )
        self.project_seed_documents = self._tag_source_kind(project_chunks, "project")
        self.seed_documents = self.curated_seed_documents + self.project_seed_documents
        self._ready = threading.Event()
        self._seed_lock = threading.Lock()
        self._background_seed_started = False

        if self.semantic_search_enabled:
            thread = threading.Thread(target=self._init_deps, daemon=True)
            thread.start()
        else:
            self._ready.set()

    def _init_deps(self) -> None:
        try:
            import numpy as np
            import ssl
            from pymongo import MongoClient
            from sentence_transformers import SentenceTransformer

            self._np = np
            print("Initializing Knowledge Engine...", flush=True)
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.client = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=5000,
                tls=True,
                tlsAllowInvalidCertificates=True,
            )
            self.db = self.client[settings.database_name]
            self.collection = self.db[settings.knowledge_collection_name]
            self.client.server_info()
            print("Knowledge Engine Ready.", flush=True)
            threading.Thread(target=self._prime_knowledge_index, daemon=True).start()
        except Exception as error:
            self.last_error = str(error)
            print(f"Knowledge Engine init failed: {error}", file=sys.stderr)
        finally:
            self._ready.set()

    def is_healthy(self) -> bool:
        if not self.semantic_search_enabled:
            return bool(self.seed_documents)

        ready = self._ready.wait(timeout=0.1)
        return ready and self.model is not None and self.collection is not None

    def warmup(self, wait_seconds: float | None = None) -> dict:
        if not self.semantic_search_enabled:
            return {
                "semantic_search_enabled": False,
                "ready": bool(self.seed_documents),
                "index_ready": bool(self.seed_documents),
                "indexed_records": len(self.seed_documents),
                "error": "",
            }

        timeout = wait_seconds if wait_seconds is not None else settings.warmup_wait_seconds
        ready = self._ready.wait(timeout=max(timeout, 0))

        if not ready or self.model is None or self.collection is None or self._np is None:
            return {
                "semantic_search_enabled": True,
                "ready": False,
                "index_ready": False,
                "error": self.last_error or "Knowledge engine is still initializing.",
            }

        try:
            self.model.encode("warmup", show_progress_bar=False)
            self._seed_if_needed()
            self._refresh_search_index(force=True)
            with self._cache_lock:
                index_ready = bool(self._indexed_records) and self._embedding_matrix is not None
                indexed_records = len(self._indexed_records)

            return {
                "semantic_search_enabled": True,
                "ready": True,
                "index_ready": index_ready,
                "indexed_records": indexed_records,
                "error": "",
            }
        except Exception as error:
            self.last_error = str(error)
            return {
                "semantic_search_enabled": True,
                "ready": False,
                "index_ready": False,
                "error": str(error),
            }

    def search(self, query: str, top_k: int = 3) -> str:
        """
        Returns only vector-retrieved TrustTrade knowledge.
        If the vector system is unavailable, return no context.
        """
        if not query:
            return ""

        if not self.semantic_search_enabled:
            return self._search_local_documents(query, top_k=top_k)

        if not self._ready.wait(timeout=settings.knowledge_init_wait_seconds):
            return ""

        if self.model is None or self.collection is None or self._np is None:
            return ""

        try:
            self._seed_if_needed()
            self._refresh_search_index()
            with self._cache_lock:
                indexed_records = list(self._indexed_records)
                embedding_matrix = self._embedding_matrix

            if not indexed_records or embedding_matrix is None:
                return ""

            query_embedding = self.model.encode(query, show_progress_bar=False)
            query_embedding = self._np.asarray(query_embedding, dtype=self._np.float32)
            if embedding_matrix.ndim != 2 or embedding_matrix.shape[1] != query_embedding.shape[0]:
                return ""
            query_norm = self._np.linalg.norm(query_embedding)
            if query_norm == 0:
                return ""
            query_embedding = query_embedding / query_norm

            query_tokens = self._tokenize(query)
            semantic_scores = embedding_matrix @ query_embedding
            similarities = []
            for index, record in enumerate(indexed_records):
                semantic_score = float(semantic_scores[index])
                title = record.get("title", "")
                text = record.get("sourceText", "")
                title_tokens = record.get("title_tokens", set())
                text_tokens = record.get("tokens", set())
                overlap = len(query_tokens & text_tokens)
                title_overlap = len(query_tokens & title_tokens)
                hybrid_score = semantic_score + (title_overlap * 0.18) + (overlap * 0.05)
                similarities.append((hybrid_score, semantic_score, record))

            if not similarities:
                return ""

            similarities.sort(key=lambda item: (item[0], item[1]), reverse=True)
            context_parts = []
            for hybrid_score, semantic_score, match in similarities[: max(top_k * 4, 8)]:
                if semantic_score <= 0.12 and hybrid_score <= 0.22:
                    continue
                title = match.get("title", "Knowledge Chunk")
                text = match.get("sourceText", "")
                if text:
                    context_parts.append(self._format_context_entry(title, text))
                if len(context_parts) >= top_k:
                    break

            return "\n\n".join(context_parts)
        except Exception as error:
            print(f"Knowledge Search Error: {error}", file=sys.stderr)
            return ""

    def _search_local_documents(self, query: str, top_k: int = 3) -> str:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return ""

        candidates: list[tuple[float, dict]] = []
        for document in self.seed_documents:
            title = document.get("title", "")
            text = document.get("sourceText", "")
            if not text:
                continue

            text_tokens = set(document.get("tokens") or self._tokenize(text))
            title_tokens = self._tokenize(title)
            overlap = len(query_tokens & text_tokens)
            title_overlap = len(query_tokens & title_tokens)
            if overlap == 0 and title_overlap == 0:
                continue

            score = float(title_overlap * 2.0 + overlap * 0.75)
            if score <= 0:
                continue

            candidates.append((score, document))

        if not candidates:
            return ""

        candidates.sort(key=lambda item: item[0], reverse=True)
        context_parts = []
        for _, match in candidates[: max(top_k * 3, 6)]:
            title = match.get("title", "Knowledge Chunk")
            text = match.get("sourceText", "")
            if not text:
                continue
            formatted = self._format_context_entry(title, text)
            if formatted not in context_parts:
                context_parts.append(formatted)
            if len(context_parts) >= top_k:
                break

        return "\n\n".join(context_parts)

    def _seed_if_needed(self) -> None:
        if self.model is None or self.collection is None:
            return

        try:
            if not self._curated_seed_checked:
                self._seed_documents(self.curated_seed_documents)
                self._curated_seed_checked = True
            if not self._background_seed_started:
                self._background_seed_started = True
                thread = threading.Thread(target=self._seed_project_documents, daemon=True)
                thread.start()
        except Exception as error:
            print(f"Knowledge seeding check skipped: {error}", file=sys.stderr)

    def seed_knowledge(self) -> int:
        return self._seed_documents(self.seed_documents)

    def _seed_project_documents(self) -> None:
        try:
            self._seed_documents(self.project_seed_documents)
            self._refresh_search_index(force=True)
        except Exception as error:
            print(f"Background project seeding failure: {error}", file=sys.stderr)

    def _seed_documents(self, documents: list[dict]) -> int:
        if self.model is None or self.collection is None:
            return 0

        count = 0
        with self._seed_lock:
            for document in documents:
                try:
                    source_text = document.get("sourceText", "")
                    title = document.get("title", "Knowledge")
                    if not source_text:
                        continue
                    stable_id = self._document_id(document)

                    existing = self.collection.find_one(
                        {
                            "$or": [
                                {"id": stable_id},
                                {"sourceText": source_text},
                            ]
                        },
                        {"_id": 1, "id": 1},
                    )
                    if existing:
                        updates = {}
                        if existing.get("id") != stable_id:
                            updates["id"] = stable_id
                        updates["title"] = title
                        updates["tokens"] = sorted(self._tokenize(f"{title} {source_text}"))
                        updates["metadata"] = self._build_seed_metadata(document)
                        if updates:
                            self.collection.update_one({"_id": existing["_id"]}, {"$set": updates})
                        continue

                    payload = self._build_seed_payload(document)
                    if not payload:
                        continue

                    self.collection.update_one(
                        {"id": stable_id},
                        {"$setOnInsert": payload},
                        upsert=True,
                    )
                    count += 1
                except Exception as error:
                    print(f"Knowledge seeding skipped one document: {error}", file=sys.stderr)

        if count:
            self._mark_index_dirty()
        return count

    def _prime_knowledge_index(self) -> None:
        try:
            self._seed_if_needed()
            self._refresh_search_index(force=True)
        except Exception as error:
            print(f"Knowledge index prime failed: {error}", file=sys.stderr)

    def _mark_index_dirty(self) -> None:
        with self._cache_lock:
            self._index_dirty = True

    def _refresh_search_index(self, force: bool = False) -> None:
        if self.collection is None or self._np is None:
            return

        with self._cache_lock:
            if not force and not self._index_dirty and self._embedding_matrix is not None:
                return

        records = list(
            self.collection.find(
                {"embedding": {"$exists": True}},
                {"title": 1, "sourceText": 1, "embedding": 1, "tokens": 1},
            )
        )

        indexed_records = []
        normalized_embeddings = []
        for record in records:
            embedding = record.get("embedding")
            if not embedding:
                continue

            vector = self._np.asarray(embedding, dtype=self._np.float32)
            if self.embedding_dim and vector.shape[0] != self.embedding_dim:
                continue
            norm = self._np.linalg.norm(vector)
            if norm == 0:
                continue

            title = record.get("title", "")
            text = record.get("sourceText", "")
            tokens = set(record.get("tokens") or self._tokenize(f"{title} {text}"))
            title_tokens = self._tokenize(title)

            indexed_records.append(
                {
                    "title": title,
                    "sourceText": text,
                    "tokens": tokens,
                    "title_tokens": title_tokens,
                }
            )
            normalized_embeddings.append(vector / norm)

        matrix = None
        if normalized_embeddings:
            matrix = self._np.vstack(normalized_embeddings)

        with self._cache_lock:
            self._indexed_records = indexed_records
            self._embedding_matrix = matrix
            self._index_dirty = False

    def _build_seed_payload(self, document: dict) -> dict | None:
        source_text = document.get("sourceText", "")
        title = document.get("title", "Knowledge")
        if not source_text:
            return None

        stable_id = self._document_id(document)
        metadata = self._build_seed_metadata(document)

        return {
            "id": stable_id,
            "title": title,
            "sourceText": source_text,
            "embedding": self.model.encode(source_text, show_progress_bar=False).tolist(),
            "tokens": sorted(self._tokenize(f"{title} {source_text}")),
            "metadata": metadata,
        }

    def _document_id(self, document: dict) -> str:
        source_text = document.get("sourceText", "")
        title = document.get("title", "Knowledge")
        return str(
            document.get("id")
            or hashlib.sha1(f"{title}\n{source_text}".encode("utf-8")).hexdigest()
        )

    def _build_seed_metadata(self, document: dict) -> dict:
        metadata = dict(document.get("metadata", {}))
        metadata.update(
            {
                "type": "platform_docs",
                "seedVersion": self.seed_version,
                "sourceKind": document.get("sourceKind", "unknown"),
            }
        )
        return metadata

    def _load_local_documents(self) -> list[dict]:
        data_dir = Path(__file__).resolve().parents[1] / "data"
        documents = []

        for path in sorted(data_dir.glob("*.txt")):
            try:
                text = path.read_text(encoding="utf-8").strip()
            except Exception as error:
                print(f"Failed to load local knowledge file {path.name}: {error}", file=sys.stderr)
                continue

            if not text:
                continue

            documents.append(
                {
                    "title": path.stem,
                    "sourceText": text,
                    "tokens": self._tokenize(text),
                }
            )

        return documents

    def _load_project_chunks(self) -> list[dict]:
        documents = []

        for record in load_project_records():
            source_text = record.get("sourceText", "").strip()
            if not source_text:
                continue

            documents.append(
                {
                    "title": record.get("title", "Project File"),
                    "sourceText": source_text,
                    "tokens": self._tokenize(source_text),
                }
            )

        return documents

    def _tag_source_kind(self, documents: list[dict], source_kind: str) -> list[dict]:
        tagged = []

        for document in documents:
            tagged.append(
                {
                    **document,
                    "sourceKind": source_kind,
                }
            )

        return tagged

    def _build_local_chunks(self, documents: list[dict]) -> list[dict]:
        chunks = []

        for document in documents:
            paragraphs = [
                paragraph.strip()
                for paragraph in re.split(r"\n\s*\n", document["sourceText"])
                if paragraph.strip()
            ]

            for paragraph in paragraphs:
                cleaned_lines = []
                for raw_line in paragraph.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    line = re.sub(r"^\d+\.\s*", "", line)
                    line = re.sub(r"^-\s*", "", line)
                    cleaned_lines.append(line)

                if not cleaned_lines:
                    continue

                chunk_text = " ".join(cleaned_lines)
                if len(chunk_text) < 40:
                    continue

                chunks.append(
                    {
                        "title": document["title"],
                        "sourceText": chunk_text,
                        "tokens": self._tokenize(chunk_text),
                    }
                )

        return chunks or documents

    def _format_context_entry(self, title: str, text: str, max_chars: int = 900) -> str:
        normalized = re.sub(r"\s+", " ", text).strip()
        truncated = normalized[:max_chars].rstrip()
        if len(normalized) > max_chars:
            last_break = max(truncated.rfind(". "), truncated.rfind("; "), truncated.rfind(", "))
            if last_break > 120:
                truncated = truncated[: last_break + 1].rstrip()
            truncated = f"{truncated} ..."
        return f"--- {title} ---\n{truncated}"

    def _tokenize(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"[a-z0-9]+", text.lower())
            if token not in STOPWORDS and len(token) > 1
        }

    def __del__(self):
        try:
            if self.client:
                self.client.close()
        except Exception:
            pass
