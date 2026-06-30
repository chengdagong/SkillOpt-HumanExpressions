import json
from pathlib import Path

import pytest

from scripts.materialize_human_expressions import materialize_human_expressions
from skillopt.envs.human_expressions.dataloader import HumanExpressionsDataLoader
from skillopt.envs.human_expressions.evaluator import evaluate_rewrite


def _write_markdown(path: Path, *, split: str, body: str) -> None:
    path.write_text(
        "\n".join(
            [
                "---",
                "id: sample",
                f"split: {split}",
                "---",
                "# 标题",
                "",
                body,
            ]
        ),
        encoding="utf-8",
    )


def test_evaluate_rewrite_scores_identical_text_as_perfect() -> None:
    result = evaluate_rewrite("她在凌晨时分到来。", "她在凌晨时分到来。")

    assert result["hard"] == 1
    assert result["soft"] == pytest.approx(1.0)
    assert result["metrics"]["char_ngram_fscore"] == pytest.approx(1.0)
    assert result["metrics"]["normalized_edit_similarity"] == pytest.approx(1.0)
    assert result["metrics"]["rouge_l_fscore"] == pytest.approx(1.0)


def test_evaluate_rewrite_uses_partial_similarity_without_ai_judge() -> None:
    result = evaluate_rewrite("她在清晨到来。", "她在凌晨时分到来。")

    assert 0.0 < result["soft"] < 1.0
    assert result["hard"] == 0
    assert set(result["metrics"]) == {
        "char_ngram_fscore",
        "normalized_edit_similarity",
        "rouge_l_fscore",
        "structure_similarity",
    }


def test_dataloader_loads_human_original_from_markdown_path(tmp_path: Path) -> None:
    split_dir = tmp_path / "split"
    train_dir = split_dir / "train"
    val_dir = split_dir / "val"
    test_dir = split_dir / "test"
    for directory in (train_dir, val_dir, test_dir):
        directory.mkdir(parents=True)

    _write_markdown(train_dir / "sample.md", split="train", body="她在凌晨时分到来。")
    item = {
        "id": "sample",
        "source_id": "sample",
        "human_original_path": "train/sample.md",
        "ai_flavored_text": "她的到来标志着命运的转折。",
    }
    for directory, items in (
        (train_dir, [item]),
        (val_dir, []),
        (test_dir, []),
    ):
        (directory / "items.json").write_text(json.dumps(items), encoding="utf-8")

    loader = HumanExpressionsDataLoader(split_dir=str(split_dir), split_mode="split_dir")
    loader.setup({"split_dir": str(split_dir), "split_mode": "split_dir"})

    assert loader.train_items[0]["human_original"] == "# 标题\n\n她在凌晨时分到来。"
    assert loader.train_items[0]["task_type"] == "human_expressions"


def test_dataloader_rejects_items_without_ai_flavored_text(tmp_path: Path) -> None:
    split_dir = tmp_path / "split"
    for split in ("train", "val", "test"):
        directory = split_dir / split
        directory.mkdir(parents=True)
        (directory / "items.json").write_text("[]", encoding="utf-8")
    (split_dir / "train" / "items.json").write_text(
        json.dumps([{"id": "sample", "human_original": "正文"}]),
        encoding="utf-8",
    )

    loader = HumanExpressionsDataLoader(split_dir=str(split_dir), split_mode="split_dir")

    with pytest.raises(ValueError, match="ai_flavored_text"):
        loader.setup({"split_dir": str(split_dir), "split_mode": "split_dir"})


def test_materialize_human_expressions_creates_train_val_test_scaffold(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "out"
    for split in ("train", "test"):
        (source_dir / split).mkdir(parents=True)

    train_items = []
    for idx in range(1, 4):
        item_id = f"story_{idx:03d}"
        rel_path = f"train/{item_id}.md"
        _write_markdown(source_dir / rel_path, split="train", body=f"训练正文 {idx}")
        train_items.append({"id": item_id, "path": rel_path, "split": "train", "chars": 10})
    test_item = {"id": "story_004", "path": "test/story_004.md", "split": "test", "chars": 10}
    _write_markdown(source_dir / test_item["path"], split="test", body="测试正文")

    (source_dir / "train" / "items.json").write_text(json.dumps(train_items), encoding="utf-8")
    (source_dir / "test" / "items.json").write_text(json.dumps([test_item]), encoding="utf-8")

    counts = materialize_human_expressions(source_dir, output_dir, val_count=1)

    assert counts == {"train": 2, "val": 1, "test": 1}
    val_items = json.loads((output_dir / "val" / "items.json").read_text(encoding="utf-8"))
    assert val_items[0]["id"] == "story_003_chunk_001"
    assert val_items[0]["ai_flavored_text"] == ""
    assert val_items[0]["human_original_path"] == "val/story_003_chunk_001.md"
    assert (output_dir / "val" / "story_003_chunk_001.md").exists()


def test_materialize_human_expressions_can_split_markdown_into_chunks(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "out"
    for split in ("train", "test"):
        (source_dir / split).mkdir(parents=True)

    body = "\n\n".join(
        [
            "第一段很短。",
            "第二段会稍微长一点，用来和前后段落组成第一个片段。",
            "第三段继续推进场景，让累计长度超过目标值。",
            "第四段应该进入后面的片段。",
            "第五段收尾。",
        ]
    )
    _write_markdown(source_dir / "train/story_001.md", split="train", body=body)
    _write_markdown(source_dir / "train/story_002.md", split="train", body="验证正文。")
    _write_markdown(source_dir / "test/story_003.md", split="test", body="测试正文。")
    (source_dir / "train" / "items.json").write_text(
        json.dumps(
            [
                {"id": "story_001", "path": "train/story_001.md", "split": "train", "chars": len(body)},
                {"id": "story_002", "path": "train/story_002.md", "split": "train", "chars": 5},
            ]
        ),
        encoding="utf-8",
    )
    (source_dir / "test" / "items.json").write_text(
        json.dumps([{"id": "story_003", "path": "test/story_003.md", "split": "test", "chars": 5}]),
        encoding="utf-8",
    )

    counts = materialize_human_expressions(
        source_dir,
        output_dir,
        val_count=1,
        chunk_target_chars=35,
        chunk_max_chars=55,
        chunk_min_chars=10,
    )

    assert counts["train"] > 1
    train_items = json.loads((output_dir / "train" / "items.json").read_text(encoding="utf-8"))
    assert [item["source_id"] for item in train_items] == ["story_001", "story_001"]
    assert [item["chunk_index"] for item in train_items] == [1, 2]
    assert train_items[0]["id"] == "story_001_chunk_001"
    assert train_items[0]["human_original_path"] == "train/story_001_chunk_001.md"
    assert (output_dir / train_items[0]["human_original_path"]).exists()

    val_items = json.loads((output_dir / "val" / "items.json").read_text(encoding="utf-8"))
    assert val_items[0]["source_id"] == "story_002"
    test_items = json.loads((output_dir / "test" / "items.json").read_text(encoding="utf-8"))
    assert test_items[0]["source_id"] == "story_003"


def test_materialize_human_expressions_can_select_explicit_val_sources(tmp_path: Path) -> None:
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "out"
    for split in ("train", "test"):
        (source_dir / split).mkdir(parents=True)

    train_items = []
    for idx in range(1, 4):
        item_id = f"story_{idx:03d}"
        rel_path = f"train/{item_id}.md"
        _write_markdown(
            source_dir / rel_path,
            split="train",
            body=f"第 {idx} 个源章节。这里有对话，也有叙述。",
        )
        train_items.append({"id": item_id, "path": rel_path, "split": "train", "chars": 20})
    _write_markdown(source_dir / "test/story_004.md", split="test", body="测试正文。")

    (source_dir / "train" / "items.json").write_text(json.dumps(train_items), encoding="utf-8")
    (source_dir / "test" / "items.json").write_text(
        json.dumps([{"id": "story_004", "path": "test/story_004.md", "split": "test", "chars": 5}]),
        encoding="utf-8",
    )

    counts = materialize_human_expressions(
        source_dir,
        output_dir,
        val_count=1,
        val_source_ids=["story_001"],
        chunk_target_chars=100,
        chunk_max_chars=120,
        chunk_min_chars=10,
    )

    assert counts == {"train": 2, "val": 1, "test": 1}
    train_items_out = json.loads((output_dir / "train" / "items.json").read_text(encoding="utf-8"))
    val_items_out = json.loads((output_dir / "val" / "items.json").read_text(encoding="utf-8"))
    assert {item["source_id"] for item in train_items_out} == {"story_002", "story_003"}
    assert [item["source_id"] for item in val_items_out] == ["story_001"]
