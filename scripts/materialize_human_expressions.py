"""Materialize Human Expressions split scaffolds from human-original Markdown data."""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE_DIR = PROJECT_ROOT / "data" / "human_expressions_witcher_volume1"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "human_expressions_witcher_volume1_chunks"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=DEFAULT_SOURCE_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-count", type=int, default=1)
    parser.add_argument(
        "--val-source-ids",
        default="",
        help="Comma-separated source item IDs to move from source train into val. Overrides --val-count when set.",
    )
    parser.add_argument("--chunk-target-chars", type=int, default=900)
    parser.add_argument("--chunk-max-chars", type=int, default=1200)
    parser.add_argument("--chunk-min-chars", type=int, default=400)
    return parser.parse_args()


def _load_items(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        data = json.load(file)
    if not isinstance(data, list):
        raise ValueError(f"Expected JSON array in {path}")
    return data


def _strip_front_matter(markdown: str) -> str:
    text = markdown.replace("\r\n", "\n").replace("\r", "\n")
    if not text.startswith("---\n"):
        return text.strip()
    end = text.find("\n---\n", 4)
    if end == -1:
        return text.strip()
    return text[end + len("\n---\n") :].strip()


def _visible_len(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _split_long_paragraph(paragraph: str, max_chars: int) -> list[str]:
    if _visible_len(paragraph) <= max_chars:
        return [paragraph]
    sentences = re.findall(r"[^。！？!?；;]+[。！？!?；;]?", paragraph)
    chunks: list[str] = []
    current = ""
    for sentence in sentences or [paragraph]:
        candidate = f"{current}{sentence}" if current else sentence
        if current and _visible_len(candidate) > max_chars:
            chunks.append(current.strip())
            current = sentence
        else:
            current = candidate
    if current.strip():
        chunks.append(current.strip())
    return chunks


def _chunk_markdown_body(
    body: str,
    *,
    target_chars: int,
    max_chars: int,
    min_chars: int,
) -> list[str]:
    paragraphs: list[str] = []
    for paragraph in re.split(r"\n\s*\n", body.strip()):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        paragraphs.extend(_split_long_paragraph(paragraph, max_chars))

    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for paragraph in paragraphs:
        paragraph_len = _visible_len(paragraph)
        candidate_len = current_len + paragraph_len
        if current and current_len >= min_chars and candidate_len > max_chars:
            chunks.append("\n\n".join(current).strip())
            current = [paragraph]
            current_len = paragraph_len
            continue
        current.append(paragraph)
        current_len = candidate_len
        if current_len >= target_chars:
            chunks.append("\n\n".join(current).strip())
            current = []
            current_len = 0
    if current:
        if chunks and current_len < min_chars:
            chunks[-1] = f"{chunks[-1]}\n\n" + "\n\n".join(current).strip()
        else:
            chunks.append("\n\n".join(current).strip())
    return chunks


def _chunk_item(
    source_dir: Path,
    output_dir: Path,
    item: dict,
    split: str,
    *,
    target_chars: int,
    max_chars: int,
    min_chars: int,
) -> list[dict]:
    source_rel = Path(str(item["path"]))
    source_path = source_dir / source_rel
    body = _strip_front_matter(source_path.read_text(encoding="utf-8"))
    chunks = _chunk_markdown_body(
        body,
        target_chars=target_chars,
        max_chars=max_chars,
        min_chars=min_chars,
    )
    out_items: list[dict] = []
    source_id = str(item["id"])
    for idx, chunk in enumerate(chunks, start=1):
        chunk_id = f"{source_id}_chunk_{idx:03d}"
        target_rel = Path(split) / f"{chunk_id}.md"
        target_path = output_dir / target_rel
        target_path.parent.mkdir(parents=True, exist_ok=True)
        target_path.write_text(chunk + "\n", encoding="utf-8")

        out = dict(item)
        out["id"] = chunk_id
        out["source_id"] = source_id
        out["split"] = split
        out["chunk_index"] = idx
        out["chunk_count"] = len(chunks)
        out["human_original_path"] = target_rel.as_posix()
        out["path"] = target_rel.as_posix()
        out["chars"] = _visible_len(chunk)
        out["ai_flavored_text"] = out.get("ai_flavored_text", "")
        out.setdefault("task_type", "human_expressions")
        out_items.append(out)
    return out_items


def materialize_human_expressions(
    source_dir: Path,
    output_dir: Path,
    *,
    val_count: int = 1,
    val_source_ids: list[str] | None = None,
    chunk_target_chars: int = 900,
    chunk_max_chars: int = 1200,
    chunk_min_chars: int = 400,
) -> dict[str, int]:
    source_dir = source_dir.resolve()
    output_dir = output_dir.resolve()
    train_items = _load_items(source_dir / "train" / "items.json")
    test_items = _load_items(source_dir / "test" / "items.json")

    if val_count < 0:
        raise ValueError("val_count must be non-negative")
    if val_count >= len(train_items):
        raise ValueError("val_count must be smaller than the number of train items")

    if val_source_ids:
        wanted_val_ids = {str(item_id) for item_id in val_source_ids}
        available_ids = {str(item["id"]) for item in train_items}
        missing = sorted(wanted_val_ids - available_ids)
        if missing:
            raise ValueError(f"val_source_ids not found in source train split: {', '.join(missing)}")
        train_source = [item for item in train_items if str(item["id"]) not in wanted_val_ids]
        val_source = [item for item in train_items if str(item["id"]) in wanted_val_ids]
    else:
        train_source = train_items[: len(train_items) - val_count] if val_count else list(train_items)
        val_source = train_items[len(train_items) - val_count :] if val_count else []

    split_items = {
        "train": [
            chunk
            for item in train_source
            for chunk in _chunk_item(
                source_dir,
                output_dir,
                item,
                "train",
                target_chars=chunk_target_chars,
                max_chars=chunk_max_chars,
                min_chars=chunk_min_chars,
            )
        ],
        "val": [
            chunk
            for item in val_source
            for chunk in _chunk_item(
                source_dir,
                output_dir,
                item,
                "val",
                target_chars=chunk_target_chars,
                max_chars=chunk_max_chars,
                min_chars=chunk_min_chars,
            )
        ],
        "test": [
            chunk
            for item in test_items
            for chunk in _chunk_item(
                source_dir,
                output_dir,
                item,
                "test",
                target_chars=chunk_target_chars,
                max_chars=chunk_max_chars,
                min_chars=chunk_min_chars,
            )
        ],
    }

    counts = {split: len(items) for split, items in split_items.items()}
    for split, items in split_items.items():
        split_path = output_dir / split
        split_path.mkdir(parents=True, exist_ok=True)
        with (split_path / "items.json").open("w", encoding="utf-8") as file:
            json.dump(items, file, ensure_ascii=False, indent=2)

    manifest = {
        "dataset": "human_expressions",
        "source_dir": str(source_dir),
        "split_method": (
            "Chunked Markdown by paragraph; copied source test split; "
            + (
                f"moved explicit source train item(s) to val: {', '.join(val_source_ids)}."
                if val_source_ids
                else f"moved last {val_count} source train item(s) to val."
            )
        ),
        "val_source_ids": val_source_ids or [str(item["id"]) for item in val_source],
        "chunking": {
            "target_chars": chunk_target_chars,
            "max_chars": chunk_max_chars,
            "min_chars": chunk_min_chars,
        },
        "counts": counts,
        "item_fields": [
            "id",
            "source_id",
            "chunk_index",
            "chunk_count",
            "chapter_index",
            "chapter_title",
            "split",
            "human_original_path",
            "ai_flavored_text",
        ],
    }
    with (output_dir / "split_manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    return counts


def main() -> None:
    args = parse_args()
    val_source_ids = [
        item.strip()
        for item in str(args.val_source_ids or "").split(",")
        if item.strip()
    ]
    counts = materialize_human_expressions(
        args.source_dir,
        args.output_dir,
        val_count=args.val_count,
        val_source_ids=val_source_ids or None,
        chunk_target_chars=args.chunk_target_chars,
        chunk_max_chars=args.chunk_max_chars,
        chunk_min_chars=args.chunk_min_chars,
    )
    print(f"Wrote Human Expressions scaffold to {args.output_dir.resolve()}: {counts}")


if __name__ == "__main__":
    main()
