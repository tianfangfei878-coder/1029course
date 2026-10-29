import csv
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.io import to_html
from qhchina import load_fonts

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

COLLECTIONS = ["呐喊", "彷徨", "故事新编", "沈从文小说集"]
COLORS = {"呐喊": "#1f77b4", "彷徨": "#d62728", "故事新编": "#2ca02c", "沈从文小说集": "#9467bd"}


def main() -> None:
    font = load_fonts()
    print(f"Loaded font: {font}")

    rows = list(csv.DictReader((ROOT / "story_coordinates.csv").open(encoding="utf-8-sig")))
    for r in rows:
        r["PC1"] = float(r["PC1"])
        r["PC2"] = float(r["PC2"])

    fig, ax = plt.subplots(figsize=(10, 8), dpi=150)
    for coll in COLLECTIONS:
        pts = [(r["PC1"], r["PC2"]) for r in rows if r["collection"] == coll]
        xs, ys = zip(*pts)
        ax.scatter(xs, ys, c=COLORS[coll], label=coll, s=140, alpha=0.85, edgecolors="black", linewidths=0.5)
        for r in (r for r in rows if r["collection"] == coll):
            label = Path(r["source_file"]).stem
            ax.annotate(label, (r["PC1"], r["PC2"]), fontsize=7, alpha=0.75)

    author_centroids = {
        author: (
            sum(r["PC1"] for r in rows if r["author"] == author) / len([r for r in rows if r["author"] == author]),
            sum(r["PC2"] for r in rows if r["author"] == author) / len([r for r in rows if r["author"] == author]),
        )
        for author in ["鲁迅", "沈从文"]
    }
    ax.scatter(
        [x for x, _ in author_centroids.values()],
        [y for _, y in author_centroids.values()],
        marker="X", s=420, c="black", label="作者重心 (author centroid)", zorder=5,
    )
    for author, (x, y) in author_centroids.items():
        ax.annotate(author, (x, y), fontsize=11, fontweight="bold",
                    xytext=(8, 8), textcoords="offset points")

    for coll in COLLECTIONS:
        coll_rows = [r for r in rows if r["collection"] == coll]
        cx = sum(r["PC1"] for r in coll_rows) / len(coll_rows)
        cy = sum(r["PC2"] for r in coll_rows) / len(coll_rows)
        ax.scatter([cx], [cy], marker="D", s=260, c=COLORS[coll],
                   edgecolors="black", linewidths=1.5, zorder=4,
                   label=f"{coll} 重心")
        ax.annotate(f"{coll}重心", (cx, cy), fontsize=8,
                    xytext=(0, 13), textcoords="offset points", ha="center", zorder=6)

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Stylometry PCA — 鲁迅 vs 沈从文 (100 features, cosine)")
    ax.legend(title="Collection")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    png_path = OUT / "pca_stories.png"
    fig.savefig(png_path)
    plt.close(fig)
    print(f"Wrote {png_path}")

    traces = []
    for coll in COLLECTIONS:
        pts = [r for r in rows if r["collection"] == coll]
        traces.append(go.Scatter(
            x=[r["PC1"] for r in pts],
            y=[r["PC2"] for r in pts],
            mode="markers+text",
            name=coll,
            text=[Path(r["source_file"]).stem for r in pts],
            textposition="top center",
            textfont=dict(size=9),
            marker=dict(color=COLORS[coll], size=13, line=dict(width=0.5, color="black")),
            customdata=[[r["document_id"], r["author"], r["collection"], r["source_file"], r["token_count"]] for r in pts],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "作者: %{customdata[1]}<br>"
                "文集: %{customdata[2]}<br>"
                "文件: %{customdata[3]}<br>"
                "词元数: %{customdata[4]}<br>"
                "PC1: %{x:.3f}  PC2: %{y:.3f}"
                "<extra></extra>"
            ),
        ))

    for author in ["鲁迅", "沈从文"]:
        pts = [r for r in rows if r["author"] == author]
        traces.append(go.Scatter(
            x=[sum(r["PC1"] for r in pts) / len(pts)],
            y=[sum(r["PC2"] for r in pts) / len(pts)],
            mode="markers+text",
            name=f"{author} 重心 (author centroid)",
            text=[f"{author}重心"],
            textposition="top right",
            textfont=dict(size=12, color="black"),
            marker=dict(symbol="x", size=20, color="black", line=dict(width=2)),
            customdata=[[author, f"{len(pts)} stories"]],
            hovertemplate=("<b>%{customdata[0]} 重心</b><br>"
                           "n = %{customdata[1]}<br>"
                           "PC1: %{x:.3f}  PC2: %{y:.3f}<extra></extra>"),
        ))

    for coll in COLLECTIONS:
        pts = [r for r in rows if r["collection"] == coll]
        traces.append(go.Scatter(
            x=[sum(r["PC1"] for r in pts) / len(pts)],
            y=[sum(r["PC2"] for r in pts) / len(pts)],
            mode="markers+text",
            name=f"{coll} 重心",
            text=[f"{coll}重心"],
            textposition="bottom center",
            textfont=dict(size=10, color="black"),
            marker=dict(symbol="diamond", size=15, color=COLORS[coll],
                        line=dict(width=1.5, color="black")),
            customdata=[[coll, f"{len(pts)} stories"]],
            hovertemplate=("<b>%{customdata[0]} 重心</b><br>"
                           "n = %{customdata[1]}<br>"
                           "PC1: %{x:.3f}  PC2: %{y:.3f}<extra></extra>"),
        ))

    pfig = go.Figure(data=traces)
    pfig.update_layout(
        title="Stylometry PCA — 鲁迅 vs 沈从文 (100 features, cosine distance)",
        xaxis_title="PC1",
        yaxis_title="PC2",
        hovermode="closest",
        legend=dict(title="文集 (click to toggle)"),
        template="plotly_white",
    )
    html_path = OUT / "pca_stories.html"
    html = to_html(
        pfig,
        include_plotlyjs="inline",
        full_html=True,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["select2d", "lasso2d"]},
        div_id="pca-plot",
    )
    html_path.write_text(html, encoding="utf-8")
    print(f"Wrote {html_path} ({html_path.stat().st_size / 1e6:.1f} MB, plotly.js embedded)")


if __name__ == "__main__":
    main()
