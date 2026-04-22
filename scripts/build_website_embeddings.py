from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Iterable, List

import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

OUTPUT_PATH = ROOT_DIR / 'apps' / 'chat_service' / 'data' / 'website_embeddings.json'
VECTOR_SIZE = 192

from apps.chat_service.data.project_index import load_project_records


def tokenize(text: str) -> List[str]:
    return re.findall(r'[a-z0-9]+', text.lower())


def text_to_vector(text: str, vector_size: int = VECTOR_SIZE) -> List[float]:
    vector = [0.0] * vector_size
    tokens = tokenize(text)

    if not tokens:
        return vector

    for token in tokens:
        index = hash(token) % vector_size
        vector[index] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector

    return [round(value / norm, 6) for value in vector]


def build_embedding_text(record: dict) -> str:
    return "\n".join(
        part
        for part in (
            record.get("title", ""),
            record.get("path", ""),
            record.get("sourceText", ""),
        )
        if part
    )


def main() -> None:
    sections: Iterable[dict] = load_project_records()
    records = []

    for section in sections:
        text = build_embedding_text(section)
        records.append({
            'id': section.get('id'),
            'title': section.get('title'),
            'path': section.get('path', ''),
            'source_text': text,
            'embedding': text_to_vector(text)
        })

    OUTPUT_PATH.write_text(json.dumps(records, indent=2))


if __name__ == '__main__':
    main()
