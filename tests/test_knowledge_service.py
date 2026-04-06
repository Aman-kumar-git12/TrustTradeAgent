from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from app.services.knowledge_service import KnowledgeService


class KnowledgeServiceTests(unittest.TestCase):
    def test_groq_only_mode_uses_local_documents_without_loading_embeddings(self) -> None:
        service = KnowledgeService()

        self.assertTrue(service.is_healthy())
        warmup = service.warmup()
        self.assertTrue(warmup["ready"])
        self.assertGreater(warmup["indexed_records"], 0)
        self.assertIsNone(service.model)

        context = service.search("How does the TrustTrade marketplace work?")
        self.assertTrue(context)
        self.assertIn("TrustTrade", context)


if __name__ == "__main__":
    unittest.main()
