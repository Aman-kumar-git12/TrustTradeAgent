from __future__ import annotations

from datetime import datetime
from typing import List, Optional
from bson import ObjectId
from pymongo import MongoClient
import sys

from ..config.settings import settings
from ..schemas.chat import ChatMessage

class HistoryService:
    def __init__(self):
        self._client = None
        self._db = None
        self._collection = None
        self._last_retry_time = 0

    @property
    def collection(self):
        if self._collection is None:
            now = datetime.utcnow().timestamp()
            # Prevent hammering the DB if it's down (60 sec cooldown)
            if now - self._last_retry_time < 60:
                return None

            self._last_retry_time = now
            try:
                self._client = MongoClient(
                    settings.mongodb_uri,
                    serverSelectionTimeoutMS=2000 # Fail fast
                )
                self._db = self._client[settings.database_name]
                self._collection = self._db[settings.conversation_collection_name]
                # Trigger a probe to verify connection
                self._client.server_info()
            except Exception as e:
                print(f"❌ History DB Connection Failed: {e}", file=sys.stderr)
                self._collection = None
                return None
        return self._collection

    def get_or_create_session(self, session_id: Optional[str], user_id: str) -> str:
        if self.collection is None:
            return session_id or "local-session"

        if session_id:
            try:
                # Check if session exists
                session = self.collection.find_one({"_id": ObjectId(session_id)})
                if session:
                    return str(session["_id"])
            except Exception:
                pass

        # Create new session
        new_session = {
            "userId": user_id,
            "title": "New Conversation",
            "messages": [],
            "createdAt": datetime.utcnow(),
            "updatedAt": datetime.utcnow()
        }
        result = self.collection.insert_one(new_session)
        return str(result.inserted_id)

    def save_message(self, session_id: str, role: str, content: str):
        if self.collection is None:
            return

        try:
            self.collection.update_one(
                {"_id": ObjectId(session_id)},
                {
                    "$push": {"messages": {"role": role, "content": content, "timestamp": datetime.utcnow()}},
                    "$set": {"updatedAt": datetime.utcnow()}
                }
            )
            
            # If it's the first user message, update the title
            if role == "user":
                session = self.collection.find_one({"_id": ObjectId(session_id)})
                if session and len(session.get("messages", [])) <= 1:
                    title = content[:40] + ("..." if len(content) > 40 else "")
                    self.collection.update_one(
                        {"_id": ObjectId(session_id)},
                        {"$set": {"title": title}}
                    )
        except Exception as e:
            print(f"⚠️ Failed to save message to MongoDB: {e}", file=sys.stderr)

    def get_session_history(self, session_id: str) -> List[ChatMessage]:
        if self.collection is None:
            return []

        try:
            session = self.collection.find_one({"_id": ObjectId(session_id)})
            if not session:
                return []
            
            messages = session.get("messages", [])
            return [ChatMessage(role=m["role"], content=m["content"]) for m in messages]
        except Exception as e:
            print(f"⚠️ Failed to load history from MongoDB: {e}", file=sys.stderr)
            return []

    def list_user_sessions(self, user_id: str):
        if self.collection is None:
            return []

        try:
            sessions = self.collection.find(
                {"userId": user_id},
                {"_id": 1, "title": 1, "updatedAt": 1}
            ).sort("updatedAt", -1)
            
            return [
                {
                    "id": str(s["_id"]),
                    "title": s.get("title", "Untitled"),
                    "updatedAt": s["updatedAt"].isoformat()
                }
                for s in sessions
            ]
        except Exception as e:
            print(f"⚠️ Failed to list sessions from MongoDB: {e}", file=sys.stderr)
            return []

    def delete_session(self, session_id: str):
        if self.collection is None:
            return False

        try:
            result = self.collection.delete_one({"_id": ObjectId(session_id)})
            return result.deleted_count > 0
        except Exception as e:
            print(f"⚠️ Failed to delete session: {e}", file=sys.stderr)
            return False
