import json
import re
import sys
from pathlib import Path

import jieba

ROOT = Path(__file__).resolve().parent
CORPORA = ROOT / "corpora" / "authors"
OUT = ROOT / "stories.json"

PUNCT_RE = re.compile(r"[\W_]+", re.UNICODE)


def clean(text: str) -> str:
    return PUNCT_RE.sub(" ", text)


def keep(token: str) -> bool:
    return bool(token) and all(ch.isalnum() for ch in token)


def main() -> int:
    files = sorted(CORPORA.rglob("*.txt"))
    if not files:
        print("ERROR: no .txt files found under", CORPORA)
        return 1

    stories = []
    problems = []

    for path in files:
        rel = path.relative_to(ROOT).as_posix()
        collection = path.parent.name
        author = path.parent.parent.name
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            problems.append({"file": rel, "reason": f"read failed: {exc}"})
            continue

        cleaned = clean(text)
        tokens = [t for t in jieba.lcut(cleaned) if keep(t)]

        if not tokens:
            problems.append({"file": rel, "reason": "no tokens produced"})
            continue

        stories.append({
            "collection": collection,
            "author": author,
            "file": rel,
            "char_count": len(text),
            "token_count": len(tokens),
            "tokens": tokens,
        })

    for s in stories:
        print(f"{s['author']}\t{s['collection']}\t{s['file']}\tchars={s['char_count']}\ttokens={s['token_count']}")

    total = sum(s["token_count"] for s in stories)
    print(f"\n{len(stories)} stories, {total} tokens total")

    if problems:
        print("\nPROBLEMS:")
        for p in problems:
            print(f"  {p['file']}: {p['reason']}")

    OUT.write_text(json.dumps(stories, ensure_ascii=False), encoding="utf-8")
    print(f"\nSaved {OUT.name}")
    return 0 if not problems else 2


if __name__ == "__main__":
    sys.exit(main())
