from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parents[3]
AGENT_ROOT = Path(__file__).resolve().parents[2]
INCLUDED_ROOTS = (
    PROJECT_ROOT / "Frontend" / "src",
    PROJECT_ROOT / "Backend" / "src",
    AGENT_ROOT / "apps",
    AGENT_ROOT / "api",
    AGENT_ROOT / "shared",
    AGENT_ROOT / "tests",
    AGENT_ROOT / "scripts",
)
INCLUDED_FILES = (
    PROJECT_ROOT / "README.md",
    AGENT_ROOT / "README.md",
    AGENT_ROOT / "main.py",
    AGENT_ROOT / "run_agent.py",
)
INCLUDED_SUFFIXES = {".js", ".jsx", ".py", ".md"}
EXCLUDED_DIRS = {"node_modules", "dist", ".git", "__pycache__", "data"}


def iter_project_source_files() -> Iterable[Path]:
    seen: set[Path] = set()

    for root in INCLUDED_ROOTS:
        if not root.exists():
            continue

        for path in sorted(root.rglob("*")):
            if path in seen or not path.is_file():
                continue
            if path.suffix.lower() not in INCLUDED_SUFFIXES:
                continue
            if any(part in EXCLUDED_DIRS for part in path.parts):
                continue
            seen.add(path)
            yield path

    for path in INCLUDED_FILES:
        if path.exists() and path.is_file() and path not in seen:
            yield path


def load_project_records(
    max_chunk_chars: int = 1400,
    max_chunks_per_file: int = 8,
) -> List[dict]:
    records: List[dict] = []

    for path in iter_project_source_files():
        try:
            raw_text = path.read_text(encoding="utf-8")
        except Exception:
            continue

        relative_path = path.relative_to(PROJECT_ROOT).as_posix()
        summary = summarize_file(relative_path, raw_text)
        chunks = chunk_source_text(raw_text, max_chunk_chars=max_chunk_chars)

        for index, chunk in enumerate(chunks[:max_chunks_per_file], start=1):
            title = relative_path
            if len(chunks) > 1:
                title = f"{relative_path} (Part {index})"

            source_text = (
                f"Project file: {relative_path}. "
                f"This chunk is grounded in the repository source and describes implementation details. "
                f"{summary} "
                f"{chunk}"
            ).strip()

            records.append(
                {
                    "id": f"{relative_path.replace('/', '_').replace('.', '_')}_{index}",
                    "title": title,
                    "path": relative_path,
                    "sourceText": source_text,
                }
            )

    return records


def chunk_source_text(text: str, max_chunk_chars: int = 1400) -> List[str]:
    cleaned_lines = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        line = re.sub(r"\s+", " ", line)
        if len(line) < 2:
            continue
        cleaned_lines.append(line)

    if not cleaned_lines:
        return []

    chunks: List[str] = []
    current_lines: List[str] = []
    current_size = 0

    for line in cleaned_lines:
        projected_size = current_size + len(line) + 1
        if current_lines and projected_size > max_chunk_chars:
            chunks.append(" ".join(current_lines))
            current_lines = [line]
            current_size = len(line)
        else:
            current_lines.append(line)
            current_size = projected_size

    if current_lines:
        chunks.append(" ".join(current_lines))

    return chunks


def summarize_file(relative_path: str, raw_text: str) -> str:
    lines = raw_text.splitlines()
    symbols = extract_matches(
        lines,
        (
            re.compile(r"^\s*class\s+([A-Za-z_][A-Za-z0-9_]*)"),
            re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)"),
            re.compile(r"^\s*function\s+([A-Za-z_][A-Za-z0-9_]*)"),
            re.compile(r"^\s*const\s+([A-Za-z_][A-Za-z0-9_]*)\s*="),
        ),
    )
    routes = extract_route_matches(lines)
    api_calls = extract_api_calls(lines)

    parts = [f"Repository path: {relative_path}."]
    if symbols:
        parts.append(f"Key symbols: {', '.join(symbols[:8])}.")
    if routes:
        parts.append(f"Declared routes: {', '.join(routes[:8])}.")
    if api_calls:
        parts.append(f"Observed API calls: {', '.join(api_calls[:8])}.")

    return " ".join(parts)


def extract_matches(lines: List[str], patterns: tuple[re.Pattern[str], ...]) -> List[str]:
    matches: List[str] = []
    seen: set[str] = set()

    for line in lines:
        for pattern in patterns:
            match = pattern.search(line)
            if not match:
                continue
            value = match.group(1).strip()
            if value and value not in seen:
                seen.add(value)
                matches.append(value)

    return matches


def extract_route_matches(lines: List[str]) -> List[str]:
    routes: List[str] = []
    seen: set[str] = set()
    patterns = (
        re.compile(r"router\.(get|post|put|patch|delete)\(\s*['\"]([^'\"]+)['\"]"),
        re.compile(r"app\.use\(\s*['\"]([^'\"]+)['\"]"),
    )

    for line in lines:
        for pattern in patterns:
            match = pattern.search(line)
            if not match:
                continue

            if pattern.pattern.startswith("router"):
                value = f"{match.group(1).upper()} {match.group(2)}"
            else:
                value = f"USE {match.group(1)}"

            if value not in seen:
                seen.add(value)
                routes.append(value)

    return routes


def extract_api_calls(lines: List[str]) -> List[str]:
    calls: List[str] = []
    seen: set[str] = set()
    pattern = re.compile(r"api\.(get|post|put|patch|delete)\(\s*([^,)]+)")

    for line in lines:
        match = pattern.search(line)
        if not match:
            continue

        value = f"{match.group(1).upper()} {match.group(2).strip()}"
        if value not in seen:
            seen.add(value)
            calls.append(value)

    return calls
