"""Dataloader for Human Expressions rewrite-to-original tasks."""
from __future__ import annotations

import json
from pathlib import Path

from skillopt.datasets.base import SplitDataLoader


def strip_front_matter(markdown: str) -> str:
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("---\n"):
        return text.strip()
    end = text.find("\n---\n", 4)
    if end == -1:
        return text.strip()
    return text[end + len("\n---\n") :].strip()


class HumanExpressionsDataLoader(SplitDataLoader):
    """Load human-expression rewrite tasks from split directories."""

    def load_split_items(self, split_path: str) -> list[dict]:
        split_dir = Path(split_path)
        items_path = split_dir / "items.json"
        if not items_path.exists():
            raise FileNotFoundError(f"No items.json found in {split_path}")
        with items_path.open(encoding="utf-8") as file:
            raw_items = json.load(file)
        if not isinstance(raw_items, list):
            raise ValueError(f"Expected JSON array in {items_path}")

        loaded: list[dict] = []
        root = Path(self.split_dir)
        for raw in raw_items:
            if not isinstance(raw, dict):
                raise ValueError(f"HumanExpressions item in {items_path} must be an object")
            if "ai_flavored_text" not in raw:
                raise ValueError(f"HumanExpressions item {raw.get('id')!r} is missing ai_flavored_text")
            item = dict(raw)
            if "human_original" not in item:
                original_path = item.get("human_original_path") or item.get("path")
                if not original_path:
                    raise ValueError(f"HumanExpressions item {item.get('id')!r} is missing human_original_path")
                with (root / str(original_path)).open(encoding="utf-8") as file:
                    item["human_original"] = strip_front_matter(file.read())
            item.setdefault("source_id", item.get("id"))
            item.setdefault("task_type", "human_expressions")
            loaded.append(item)
        return loaded

