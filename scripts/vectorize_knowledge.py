import os
import sys
import glob
from typing import List, Dict
from datetime import datetime, timezone

# Add parent directory to path to handle imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pymongo import MongoClient
from sentence_transformers import SentenceTransformer
from app.config.settings import settings
from app.data.project_index import load_project_records

MONGODB_URI = settings.mongodb_uri
DATABASE_NAME = settings.database_name
COLLECTION_NAME = settings.knowledge_collection_name

def chunk_text(text: str, max_chars: int = 1500) -> List[str]:
    """Simple chunking by paragraphs or max length."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        if len(current_chunk) + len(para) < max_chars:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def vectorize():
    print("🚀 Initializing Semantic Vectorizer...")
    
    # Initialize model
    model = SentenceTransformer('all-MiniLM-L6-v2')
    
    # Connect to MongoDB
    client = MongoClient(MONGODB_URI)
    db = client[DATABASE_NAME]
    collection = db[COLLECTION_NAME]
    
    # Find all .txt files in data directory
    data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'app', 'data')
    txt_files = glob.glob(os.path.join(data_dir, "*.txt"))
    
    if not txt_files:
        print(f"❌ No .txt files found in {data_dir}")
        return

    print(f"📂 Found {len(txt_files)} text files. Clearing old knowledge...")
    collection.delete_many({}) # Clear existing records for a fresh start

    total_chunks = 0
    for file_path in txt_files:
        file_name = os.path.basename(file_path)
        title = file_name.replace(".txt", "").capitalize()
        
        print(f"📄 Processing {file_name}...")
        
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        chunks = chunk_text(content)
        
        for i, chunk in enumerate(chunks):
            # Generate embedding
            embedding = model.encode(chunk).tolist()
            
            # Prepare record
            record = {
                "id": f"{file_name.lower().replace('.', '_')}_{i}",
                "title": f"{title} (Part {i+1})",
                "summary": chunk[:200] + "...",
                "sourceText": chunk,
                "embedding": embedding,
                "createdAt": datetime.now(timezone.utc),
                "updatedAt": datetime.now(timezone.utc)
            }
            
            collection.insert_one(record)
            total_chunks += 1

    project_records = load_project_records()
    for record in project_records:
        chunk = record["sourceText"]
        embedding = model.encode(chunk).tolist()
        collection.insert_one(
            {
                "id": record["id"],
                "title": record["title"],
                "summary": chunk[:200] + ("..." if len(chunk) > 200 else ""),
                "sourceText": chunk,
                "embedding": embedding,
                "createdAt": datetime.now(timezone.utc),
                "updatedAt": datetime.now(timezone.utc),
            }
        )
        total_chunks += 1
            
    print(
        f"✅ Successfully vectorized {len(txt_files)} curated knowledge files and "
        f"{len(project_records)} project chunks into {total_chunks} semantic chunks in MongoDB."
    )
    client.close()

if __name__ == "__main__":
    vectorize()
