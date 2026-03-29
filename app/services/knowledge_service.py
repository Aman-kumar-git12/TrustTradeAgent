import sys
import threading
import re
from pathlib import Path
from ..config.settings import settings

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "for", "from",
    "explain", "help", "how", "i", "in", "is", "it", "me", "my", "of", "on", "or", "the", "to",
    "tell", "works", "working",
    "what", "when", "where", "who", "why", "with", "you", "your",
}

BRAND_TOKENS = {"trusttrade"}


class KnowledgeService:
    """
    Semantic knowledge retrieval over MongoDB-stored embeddings.

    Model loading and Mongo connection happen in a background thread so
    they never block the FastAPI startup path or the first incoming request.
    """

    def __init__(self):
        self.model = None
        self._np = None
        self.client = None
        self.db = None
        self.collection = None
        self.last_error = ''
        self.semantic_search_enabled = settings.enable_semantic_search
        self.local_documents = self._load_local_documents()
        self.local_chunks = self._build_local_chunks(self.local_documents)
        self._ready = threading.Event()  # set once init completes

        if self.semantic_search_enabled:
            t = threading.Thread(target=self._init_deps, daemon=True)
            t.start()
        else:
            self._ready.set()

    def _init_deps(self) -> None:
        """Load heavy dependencies in a background thread."""
        try:
            # Import here so the module-level import is not slow
            import numpy as np
            from pymongo import MongoClient
            from sentence_transformers import SentenceTransformer

            self._np = np
            print("🤖 Initializing Knowledge Engine (background)...", flush=True)
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

            self.client = MongoClient(
                settings.mongodb_uri,
                serverSelectionTimeoutMS=5000
            )
            self.db = self.client[settings.database_name]
            self.collection = self.db[settings.knowledge_collection_name]
            # Lightweight probe — fails fast if Mongo is unreachable
            self.client.server_info()
            print("✅ Knowledge Engine Ready.", flush=True)
        except Exception as e:
            self.last_error = str(e)
            print(f"⚠️ Knowledge Engine init failed (non-fatal): {e}", file=sys.stderr)
        finally:
            self._ready.set()  # always unblock waiters, even on failure

    def is_healthy(self) -> bool:
        """Checks if the knowledge engine is initialized and connected."""
        if not self.semantic_search_enabled:
            return bool(self.local_chunks or self.local_documents)

        # Use a short timeout so we don't block the health check itself
        ready = self._ready.wait(timeout=0.1)
        return ready and self.model is not None and self.collection is not None

    def search(self, query: str, top_k: int = 3) -> str:
        """
        Returns relevant knowledge chunks as a formatted string.
        Gives the background init only a short grace period, then returns ''
        so chat stays responsive even if embeddings or Mongo are still warming up.
        """
        if not query:
            return ""

        if not self.semantic_search_enabled:
            return self._search_local(query, top_k=top_k)

        # Keep user-facing chat snappy during cold starts or degraded environments.
        if not self._ready.wait(timeout=settings.knowledge_init_wait_seconds):
            return self._search_local(query, top_k=top_k)

        if self.model is None or self.collection is None:
            return self._search_local(query, top_k=top_k)

        try:
            np = self._np
            if np is None:
                return self._search_local(query, top_k=top_k)

            query_embedding = self.model.encode(query, show_progress_bar=False)

            records = list(self.collection.find({"embedding": {"$exists": True}}))
            if not records:
                return ""

            similarities = []
            for record in records:
                if 'embedding' not in record:
                    continue
                rec_emb = np.array(record['embedding'])
                if rec_emb.shape != query_embedding.shape:
                    continue
                score = np.dot(query_embedding, rec_emb) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(rec_emb)
                )
                similarities.append((score, record))

            if not similarities:
                return self._search_local(query, top_k=top_k)

            similarities.sort(key=lambda x: x[0], reverse=True)

            context_parts = []
            for score, match in similarities[:top_k]:
                if score > 0.3:
                    title = match.get('title', 'Knowledge Chunk')
                    text = match.get('sourceText', '')
                    if text:
                        context_parts.append(f"--- {title} ---\n{text}")

            result = "\n\n".join(context_parts)
            return result or self._search_local(query, top_k=top_k)

        except Exception as e:
            print(f"❌ Knowledge Search Error: {e}", file=sys.stderr)
            return self._search_local(query, top_k=top_k)

    def _load_local_documents(self) -> list[dict]:
        data_dir = Path(__file__).resolve().parents[1] / 'data'
        documents = []

        for path in sorted(data_dir.glob('*.txt')):
            try:
                text = path.read_text(encoding='utf-8').strip()
            except Exception as error:
                print(f"⚠️ Failed to load local knowledge file {path.name}: {error}", file=sys.stderr)
                continue

            if not text:
                continue

            documents.append({
                'title': path.stem,
                'sourceText': text,
                'tokens': self._tokenize(text)
            })

        return documents

    def _build_local_chunks(self, documents: list[dict]) -> list[dict]:
        chunks = []

        for document in documents:
            paragraphs = [
                paragraph.strip()
                for paragraph in re.split(r'\n\s*\n', document['sourceText'])
                if paragraph.strip()
            ]

            for paragraph in paragraphs:
                cleaned_lines = []
                for raw_line in paragraph.splitlines():
                    line = raw_line.strip()
                    if not line:
                        continue
                    line = re.sub(r'^\d+\.\s*', '', line)
                    line = re.sub(r'^-\s*', '', line)
                    cleaned_lines.append(line)

                if not cleaned_lines:
                    continue

                chunk_text = " ".join(cleaned_lines)
                if len(chunk_text) < 40:
                    continue

                chunks.append({
                    'title': document['title'],
                    'sourceText': chunk_text,
                    'tokens': self._tokenize(chunk_text)
                })

        return chunks or documents

    def _search_local(self, query: str, top_k: int = 3) -> str:
        if not self.local_chunks:
            return ""

        query_tokens = self._tokenize(query)
        if not query_tokens:
            return ""

        min_score = 1 if query_tokens <= BRAND_TOKENS else 2
        scored = []
        for document in self.local_chunks:
            overlap_tokens = query_tokens & document['tokens']
            meaningful_query_tokens = query_tokens - BRAND_TOKENS
            meaningful_overlap = overlap_tokens - BRAND_TOKENS
            if meaningful_query_tokens and not meaningful_overlap:
                continue

            overlap = len(overlap_tokens)
            title_tokens = self._tokenize(document['title'])
            title_boost = len(query_tokens & title_tokens) * 2
            score = overlap + title_boost
            if score >= min_score:
                scored.append((score, document))

        scored.sort(key=lambda item: item[0], reverse=True)

        context_parts = []
        for _, document in scored[:top_k]:
            context_parts.append(f"--- {document['title']} ---\n{document['sourceText']}")

        return "\n\n".join(context_parts)

    def _tokenize(self, text: str) -> set[str]:
        return {
            token
            for token in re.findall(r'[a-z0-9]+', text.lower())
            if token not in STOPWORDS and len(token) > 1
        }

    def __del__(self):
        try:
            if self.client:
                self.client.close()
        except Exception:
            pass
