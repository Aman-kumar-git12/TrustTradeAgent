from __future__ import annotations

import json
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from uuid import uuid4

from bson import ObjectId
from pymongo import MongoClient

from ..config.settings import settings
from ..schemas.chat import ChatMessage


class HistoryService:
    def __init__(self):
        self._client = None
        self._db = None
        self._collection = None
        self._last_retry_time = 0
        self._file_lock = threading.Lock()
        self._file_path = Path(__file__).resolve().parents[2] / ".local_conversations.json"

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
            return self._get_or_create_file_session(session_id, user_id)

        if session_id:
            try:
                # If session_id is a valid ObjectId, convert it, otherwise try to query it as a string
                query_id = ObjectId(session_id) if len(str(session_id)) == 24 else session_id
                # Check if session exists
                session = self.collection.find_one({"_id": query_id})
                if session:
                    return str(session["_id"])
            except Exception:
                pass
            
            # Use the provided session_id if it doesn't already exist (trust the frontend payload)
            new_id = ObjectId(session_id) if len(str(session_id)) == 24 else session_id
        else:
            new_id = ObjectId()

        # Create new session
        new_session = {
            "_id": new_id,
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
            self._save_file_message(session_id, role, content)
            return

        try:
            query_id = ObjectId(session_id) if len(str(session_id)) == 24 else session_id
            self.collection.update_one(
                {"_id": query_id},
                {
                    "$push": {"messages": {"role": role, "content": content, "timestamp": datetime.utcnow()}},
                    "$set": {"updatedAt": datetime.utcnow()}
                }
            )
            
            # If it's the first user message, update the title
            if role == "user":
                session = self.collection.find_one({"_id": query_id})
                if session and len(session.get("messages", [])) <= 1:
                    title = content[:40] + ("..." if len(content) > 40 else "")
                    self.collection.update_one(
                        {"_id": query_id},
                        {"$set": {"title": title}}
                    )
        except Exception as e:
            print(f"⚠️ Failed to save message to MongoDB: {e}", file=sys.stderr)

    def get_session_history(self, session_id: str) -> List[ChatMessage]:
        if self.collection is None:
            return self._get_file_session_history(session_id)

        try:
            query_id = ObjectId(session_id) if len(str(session_id)) == 24 else session_id
            session = self.collection.find_one({"_id": query_id})
            if not session:
                return []
            
            messages = session.get("messages", [])
            return [ChatMessage(role=m["role"], content=m["content"]) for m in messages]
        except Exception as e:
            print(f"⚠️ Failed to load history from MongoDB: {e}", file=sys.stderr)
            return []

    def list_user_sessions(self, user_id: str):
        if self.collection is None:
            return self._list_file_sessions(user_id)

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
            return self._delete_file_session(session_id)

        try:
            query_id = ObjectId(session_id) if len(str(session_id)) == 24 else session_id
            result = self.collection.delete_one({"_id": query_id})
            return result.deleted_count > 0
        except Exception as e:
            print(f"⚠️ Failed to delete session: {e}", file=sys.stderr)
            return False

    def _load_file_store(self) -> dict:
        if not self._file_path.exists():
            return {"sessions": {}}

        try:
            payload = json.loads(self._file_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"⚠️ Failed to read local history store: {e}", file=sys.stderr)
            return {"sessions": {}}

        if not isinstance(payload, dict):
            return {"sessions": {}}

        sessions = payload.get("sessions")
        if not isinstance(sessions, dict):
            payload["sessions"] = {}

        return payload

    def _write_file_store(self, payload: dict) -> None:
        self._file_path.parent.mkdir(parents=True, exist_ok=True)
        self._file_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _get_or_create_file_session(self, session_id: Optional[str], user_id: str) -> str:
        with self._file_lock:
            payload = self._load_file_store()
            sessions = payload["sessions"]

            if session_id and session_id in sessions:
                return session_id

            new_session_id = session_id or uuid4().hex
            now = datetime.utcnow().isoformat()
            sessions[new_session_id] = {
                "userId": user_id,
                "title": "New Conversation",
                "messages": [],
                "createdAt": now,
                "updatedAt": now,
            }
            self._write_file_store(payload)
            return new_session_id

    def _save_file_message(self, session_id: str, role: str, content: str) -> None:
        with self._file_lock:
            payload = self._load_file_store()
            sessions = payload["sessions"]
            session = sessions.get(session_id)

            if session is None:
                now = datetime.utcnow().isoformat()
                session = {
                    "userId": "anonymous",
                    "title": "New Conversation",
                    "messages": [],
                    "createdAt": now,
                    "updatedAt": now,
                }
                sessions[session_id] = session

            session.setdefault("messages", []).append(
                {
                    "role": role,
                    "content": content,
                    "timestamp": datetime.utcnow().isoformat(),
                }
            )
            session["updatedAt"] = datetime.utcnow().isoformat()

            if role == "user" and len(session["messages"]) <= 1:
                session["title"] = content[:40] + ("..." if len(content) > 40 else "")

            self._write_file_store(payload)

    def _get_file_session_history(self, session_id: str) -> List[ChatMessage]:
        with self._file_lock:
            payload = self._load_file_store()
            session = payload["sessions"].get(session_id)
            if not session:
                return []

            messages = session.get("messages", [])
            return [
                ChatMessage(role=str(message.get("role", "")), content=str(message.get("content", "")))
                for message in messages
                if message.get("role") and message.get("content")
            ]

    def _list_file_sessions(self, user_id: str):
        with self._file_lock:
            payload = self._load_file_store()
            sessions = payload["sessions"]
            filtered = []

            for session_id, session in sessions.items():
                if session.get("userId") != user_id:
                    continue
                filtered.append(
                    {
                        "id": session_id,
                        "title": session.get("title", "Untitled"),
                        "updatedAt": session.get("updatedAt", ""),
                    }
                )

            filtered.sort(key=lambda item: item.get("updatedAt", ""), reverse=True)
            return filtered

    def _delete_file_session(self, session_id: str) -> bool:
        with self._file_lock:
            payload = self._load_file_store()
            sessions = payload["sessions"]
            if session_id not in sessions:
                return False

            sessions.pop(session_id, None)
            self._write_file_store(payload)
            return True
