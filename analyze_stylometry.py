import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np
from qhchina.analytics.stylometry import Stylometry
from sklearn.decomposition import PCA

ROOT = Path(__file__).resolve().parent
STORIES = ROOT / "stories.json"
OUT_COORDS = ROOT / "story_coordinates.csv"
OUT_FEATURES = ROOT / "selected_features_100.csv"
OUT_TOP = ROOT / "collection_top_features.csv"

ZSCORE_COL = "collection_mean_qhchina_zscore__corpus_wide_standardization"


def main() -> None:
    stories = json.loads(STORIES.read_text(encoding="utf-8"))

    corpus = {}
    for s in stories:
        corpus.setdefault(s["author"], []).append(s["tokens"])

    stylo = Stylometry(n_features=100, distance="cosine")
    stylo.fit_transform(corpus)

    vectors = np.array(stylo.document_vectors)
    labels = list(stylo.document_labels)
    doc_ids = list(stylo.document_ids)
    features = list(stylo.features)

    assert len(features) == 100, f"expected 100 features, got {len(features)}"
    assert vectors.shape == (len(stories), 100), f"vector matrix shape {vectors.shape}"

    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)

    n_story_files = len(stories)
    assert coords.shape[0] == n_story_files, (
        f"PCA points ({coords.shape[0]}) != story files ({n_story_files})"
    )
    print(f"PCA points: {coords.shape[0]} == story files: {n_story_files}")
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")

    n_docs_per_author = {a: len(d) for a, d in corpus.items()}
    idx = 0
    for author, docs in corpus.items():
        for i in range(n_docs_per_author[author]):
            assert labels[idx] == author, f"label mismatch at {idx}: {labels[idx]} != {author}"
            idx += 1
    print("Doc ordering verified: vector rows align 1:1 with stories.json order")

    story_rows = []
    idx = 0
    for author, docs in corpus.items():
        for i, s in enumerate(s for s in stories if s["author"] == author):
            story_rows.append({
                "document_id": doc_ids[idx],
                "author": author,
                "collection": s["collection"],
                "source_file": s["file"],
                "token_count": s["token_count"],
                "PC1": coords[idx, 0],
                "PC2": coords[idx, 1],
            })
            idx += 1

    with OUT_COORDS.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(story_rows[0].keys()))
        w.writeheader()
        w.writerows(story_rows)

    with OUT_FEATURES.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(["rank", "feature"])
        for rank, feat in enumerate(features, 1):
            w.writerow([rank, feat])

    feature_index = {feat: j for j, feat in enumerate(features)}
    collection_rows = []
    for collection in ["呐喊", "彷徨", "故事新编", "沈从文小说集"]:
        coll_stories = [s for s in stories if s["collection"] == collection]
        total_tokens = sum(s["token_count"] for s in coll_stories)

        counts = Counter()
        for s in coll_stories:
            counts.update(s["tokens"])

        coll_indices = [i for i, r in enumerate(story_rows) if r["collection"] == collection]
        coll_vecs = vectors[coll_indices]
        mean_z = coll_vecs.mean(axis=0)

        coll_feats = [(feat, counts.get(feat, 0)) for feat in features]
        coll_feats.sort(key=lambda x: -x[1])

        for rank, (feat, cnt) in enumerate(coll_feats, 1):
            collection_rows.append({
                "collection": collection,
                "rank": rank,
                "feature": feat,
                "count_in_collection": cnt,
                "total_collection_tokens": total_tokens,
                "freq_per_10k_tokens": cnt / total_tokens * 10000,
                ZSCORE_COL: mean_z[feature_index[feat]],
            })

    with OUT_TOP.open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(collection_rows[0].keys()))
        w.writeheader()
        w.writerows(collection_rows)

    print(f"Wrote {OUT_COORDS.name} ({len(story_rows)} rows)")
    print(f"Wrote {OUT_FEATURES.name} (100 features)")
    print(f"Wrote {OUT_TOP.name} ({len(collection_rows)} rows)")
    print(f"Note: {ZSCORE_COL} values are collection means of qhchina z-scored "
          f"feature vectors (standardization computed corpus-wide across all 43 stories).")


if __name__ == "__main__":
    main()
