import hashlib
import re
from pathlib import Path
from typing import List, Dict

from .knowledge_service import get_knowledge_components, _tokenize
from ..data.project_index import load_project_records

def seed_knowledge_data(verbose: bool = True) -> dict:
    """
    Orchestrates the seeding of local knowledge data into MongoDB.
    """
    model, collection = get_knowledge_components()
    if model is None or collection is None:
        return {"error": "Knowledge engine or DB not initialized."}

    stats = {"processed_files": 0, "inserted_chunks": 0, "skipped_chunks": 0}
    
    # 1. Load Local Document Files
    data_dir = Path(__file__).resolve().parents[1] / "data"
    documents = []
    
    for path in sorted(data_dir.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8").strip()
            if not text: continue
            documents.append({"title": path.stem, "text": text})
            stats["processed_files"] += 1
        except Exception as e:
            if verbose: print(f"❌ Failed to load {path.name}: {e}")

    # 2. Add Project Records
    for record in load_project_records():
        text = record.get("sourceText", "").strip()
        if text:
            documents.append({"title": record.get("title", "Project"), "text": text})

    # 3. Process Chunks
    for doc in documents:
        chunks = _build_chunks(doc["text"])
        for chunk in chunks:
            stable_id = hashlib.sha1(f"{doc['title']}\n{chunk}".encode("utf-8")).hexdigest()
            
            payload = {
                "id": stable_id,
                "title": doc["title"],
                "sourceText": chunk,
                "embedding": model.encode(chunk, show_progress_bar=False).tolist(),
                "tokens": sorted(_tokenize(f"{doc['title']} {chunk}")),
                "metadata": {"type": "platform_docs"}
            }
            
            try:
                # Upsert by content hash
                result = collection.update_one(
                    {"id": stable_id},
                    {"$set": payload},
                    upsert=True
                )
                if result.upserted_id or result.modified_count:
                    stats["inserted_chunks"] += 1
                else:
                    stats["skipped_chunks"] += 1
            except Exception as e:
                if verbose: print(f"❌ Failed to insert chunk: {e}")

    return stats

def _build_chunks(text: str) -> List[str]:
    """Splits a large document into overlapping paragraph chunks."""
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    
    for p in paragraphs:
        # Pre-process cleanup
        cleaned = re.sub(r"\s+", " ", p).strip()
        if len(cleaned) < 40: continue
        chunks.append(cleaned)
        
    return chunks
