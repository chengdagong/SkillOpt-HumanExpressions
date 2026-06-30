"""Rollout for Human Expressions rewrite-to-original tasks."""
from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor

from skillopt.envs.human_expressions.evaluator import evaluate_rewrite
from skillopt.model import chat_target


def _build_system(skill_content: str) -> str:
    return (
        "You are restoring a Chinese fiction excerpt that was rewritten to sound AI-generated.\n"
        "Use the provided skill as editing guidance. Output only the restored Chinese prose.\n\n"
        f"## Skill\n{skill_content.strip()}"
    )


def _build_user(item: dict) -> str:
    instruction = item.get("instruction") or (
        "请去除下面小说片段中的 AI 味，尽量恢复成原本的人类写作版本。只输出正文。"
    )
    return f"{instruction}\n\n## AI 化版本\n{item.get('ai_flavored_text', '')}"


def process_one(
    item: dict,
    out_root: str,
    skill_content: str,
    *,
    exec_timeout: int = 120,
    max_completion_tokens: int = 16384,
) -> dict:
    item_id = str(item["id"])
    pred_dir = os.path.join(out_root, "predictions", item_id)
    os.makedirs(pred_dir, exist_ok=True)
    result = {
        "id": item_id,
        "source_id": item.get("source_id", item_id),
        "task_type": item.get("task_type", "human_expressions"),
        "hard": 0,
        "soft": 0.0,
        "predicted_text": "",
        "reference_text": item.get("human_original", ""),
        "response": "",
        "fail_reason": "",
        "agent_ok": False,
    }
    if not str(item.get("ai_flavored_text") or "").strip():
        result["fail_reason"] = "missing ai_flavored_text"
        return result

    system = _build_system(skill_content)
    user = _build_user(item)
    with open(os.path.join(pred_dir, "target_system_prompt.txt"), "w", encoding="utf-8") as file:
        file.write(system)
    with open(os.path.join(pred_dir, "target_user_prompt.txt"), "w", encoding="utf-8") as file:
        file.write(user)

    try:
        response, _usage = chat_target(
            system=system,
            user=user,
            max_completion_tokens=max_completion_tokens,
            retries=5,
            stage="rollout",
            timeout=exec_timeout,
        )
        score = evaluate_rewrite(response, item.get("human_original", ""))
        result.update(score)
        result["response"] = response
        result["agent_ok"] = True
        if not result["hard"]:
            result["fail_reason"] = f"score={result['soft']:.4f} below threshold"
        with open(os.path.join(pred_dir, "evaluation.json"), "w", encoding="utf-8") as file:
            json.dump(score, file, ensure_ascii=False, indent=2)
    except Exception as exc:  # noqa: BLE001
        result["fail_reason"] = f"error: {type(exc).__name__}: {exc}"
    return result


def run_batch(
    *,
    items: list[dict],
    out_root: str,
    skill_content: str,
    exec_timeout: int = 120,
    workers: int = 4,
    max_completion_tokens: int = 16384,
) -> list[dict]:
    os.makedirs(out_root, exist_ok=True)
    results_path = os.path.join(out_root, "results.jsonl")
    if workers <= 1 or len(items) <= 1:
        results = [
            process_one(
                item,
                out_root,
                skill_content,
                exec_timeout=exec_timeout,
                max_completion_tokens=max_completion_tokens,
            )
            for item in items
        ]
    else:
        with ThreadPoolExecutor(max_workers=min(workers, len(items))) as executor:
            futures = [
                executor.submit(
                    process_one,
                    item,
                    out_root,
                    skill_content,
                    exec_timeout=exec_timeout,
                    max_completion_tokens=max_completion_tokens,
                )
                for item in items
            ]
            results = [future.result() for future in futures]
    with open(results_path, "w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    return results

