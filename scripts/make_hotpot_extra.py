"""Sample HotpotQA validation examples and write them in `hotpot_mini.json` schema.

Dùng dataset `hotpotqa/hotpot_qa` (parquet mirror trên HuggingFace). Dataset cũ
`hotpot_qa` đã bị rename và script-loader broken sau khi `trust_remote_code`
bị gỡ.
"""
import json
from collections import Counter
from pathlib import Path

from datasets import load_dataset

# Tỉ lệ câu hỏi theo difficulty. Tổng = số mẫu cần.
LEVEL_COUNTS = {"easy": 20, "medium": 50, "hard": 30}

OUT_PATH = Path(__file__).resolve().parent.parent / "data" / "hotpot_extra.json"


def to_mini_entry(ex: dict) -> dict:
    gold_titles = set(ex["supporting_facts"]["title"])
    titles = ex["context"]["title"]
    sentences = ex["context"]["sentences"]
    context = [
        {"title": t, "text": "".join(sents).strip()}
        for t, sents in zip(titles, sentences)
        if t in gold_titles
    ]
    return {
        "qid": ex["id"],
        "difficulty": ex.get("level") or "medium",
        "question": ex["question"],
        "gold_answer": ex["answer"],
        "context": context,
    }


def main() -> None:
    # Dùng `train` vì `validation` của HotpotQA toàn level=hard (design của dataset).
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="train")
    ds = ds.shuffle(seed=42)

    dist = Counter(ex.get("level") or "medium" for ex in ds)
    print(f"Train split size={len(ds)}, level distribution={dict(dist)}")

    buckets: dict[str, list[dict]] = {lvl: [] for lvl in LEVEL_COUNTS}
    for ex in ds:
        lvl = ex.get("level") or "medium"
        if lvl in buckets and len(buckets[lvl]) < LEVEL_COUNTS[lvl]:
            buckets[lvl].append(to_mini_entry(ex))
        if all(len(buckets[l]) >= n for l, n in LEVEL_COUNTS.items()):
            break

    entries = [e for lvl in LEVEL_COUNTS for e in buckets[lvl]]
    OUT_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False))

    summary = ", ".join(
        f"{lvl}={len(buckets[lvl])}/{LEVEL_COUNTS[lvl]}" for lvl in LEVEL_COUNTS
    )
    print(f"Wrote {len(entries)} entries to {OUT_PATH}  ({summary})")
    missing = [lvl for lvl, n in LEVEL_COUNTS.items() if len(buckets[lvl]) < n]
    if missing:
        print(
            f"⚠️  Thiếu quota ở level: {missing}. "
            "Chỉnh lại LEVEL_COUNTS theo phân phối thực tế in ở trên."
        )


if __name__ == "__main__":
    main()
