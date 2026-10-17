"""Draw docs/figures/results.svg and docs/figures/ablations.svg from experiments/runs/*.json.

    python scripts/plot_results.py

Plain SVG, no plotting library, so the figures rebuild anywhere.
"""
import glob
import json
import os
import statistics

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")

MANCE, WORD, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1b2130", "#52514e", "#e3e6ec"
FONT = "-apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def load():
    runs = {}
    for path in glob.glob(os.path.join(ROOT, "experiments", "runs", "*.json")):
        r = json.load(open(path))
        runs.setdefault((r["dataset"], r["tag"]), []).append(r)
    return runs


def scores(runs, dataset, tag):
    return [r["result"]["test"][r["metric"]] for r in runs.get((dataset, tag), [])]


def drops(runs, dataset, tag, kind="rule"):
    return [r["result"][f"oov_{kind}"]["drop"] for r in runs.get((dataset, tag), [])]


def mean_sd(values):
    return statistics.mean(values), (statistics.stdev(values) if len(values) > 1 else 0.0)


def svg(width, height, body):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
            f'width="{width}" height="{height}" font-family="{FONT}">\n'
            f'<rect width="{width}" height="{height}" rx="12" fill="#ffffff"/>\n{body}</svg>\n')


def text(x, y, s, size=12.5, fill=MUTED, anchor="start", weight=400):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}" '
            f'text-anchor="{anchor}" font-weight="{weight}">{s}</text>\n')


def bar(x, y0, width, top, color):
    """Bar from baseline y0 up to y=top, rounded only at the data end."""
    r = 4
    return (f'<path d="M{x:.1f},{y0:.1f} L{x:.1f},{top + r:.1f} Q{x:.1f},{top:.1f} {x + r:.1f},{top:.1f} '
            f'L{x + width - r:.1f},{top:.1f} Q{x + width:.1f},{top:.1f} {x + width:.1f},{top + r:.1f} '
            f'L{x + width:.1f},{y0:.1f} Z" fill="{color}"/>\n')


def results_figure(runs):
    panels = [("ag_news", "AG News", "accuracy (%)"),
              ("dbpedia", "DBpedia", "accuracy (%)"),
              ("germeval14", "GermEval 2014", "entity F1")]
    width, height = 920, 340
    top, base = 92, 282
    scale = (base - top) / 100
    body = text(30, 34, "Test scores: MANCE vs a word-embedding baseline", 16, INK, weight=600)
    body += f'<rect x="30" y="48" width="12" height="12" rx="3" fill="{MANCE}"/>'
    body += text(48, 59, "MANCE (3 seeds, mean ± sd)", 12.5, INK)
    body += f'<rect x="250" y="48" width="12" height="12" rx="3" fill="{WORD}"/>'
    body += text(268, 59, "word embeddings, same encoder (1 seed)", 12.5, INK)

    for i, (dataset, name, metric) in enumerate(panels):
        left = 60 + i * 290
        for tick in (0, 25, 50, 75, 100):
            y = base - tick * scale
            body += f'<line x1="{left}" y1="{y:.1f}" x2="{left + 230}" y2="{y:.1f}" stroke="{GRID}"/>\n'
            body += text(left - 8, y + 4, tick, 11, MUTED, "end")
        for j, (tag, color) in enumerate((("mance", MANCE), ("word_baseline", WORD))):
            values = scores(runs, dataset, tag)
            if not values:
                continue
            m, sd = mean_sd(values)
            x = left + 40 + j * 86
            body += bar(x, base, 64, base - m * scale, color)
            if sd:
                y1, y2 = base - (m + sd) * scale, base - (m - sd) * scale
                body += f'<line x1="{x + 32}" y1="{y1:.1f}" x2="{x + 32}" y2="{y2:.1f}" stroke="{INK}" stroke-width="1.5"/>\n'
            body += text(x + 32, base - m * scale - 8 - sd * scale, f"{m:.1f}", 13, INK, "middle", 600)
        body += text(left + 115, base + 22, name, 13.5, INK, "middle", 600)
        body += text(left + 115, base + 40, metric, 12, MUTED, "middle")
    return svg(width, height, body)


def ablations_figure(runs):
    rows = [("mance", "MANCE (character Transformer, ordered)", MANCE),
            ("no_char_order", "no character order", INK),
            ("avg_chars", "averaged characters, no Transformer", INK),
            ("word_baseline", "word embeddings", WORD)]
    width, height = 920, 300
    body = text(30, 34, "AG News: what each part contributes", 16, INK, weight=600)

    # left panel: accuracy dot plot, axis 80-95 (dots, so the axis need not start at 0)
    lo, hi, left, right = 80, 95, 300, 560
    sx = lambda v: left + (v - lo) / (hi - lo) * (right - left)
    body += text(left, 64, "test accuracy (%)", 12.5, MUTED)
    for tick in range(lo, hi + 1, 5):
        body += f'<line x1="{sx(tick):.1f}" y1="76" x2="{sx(tick):.1f}" y2="246" stroke="{GRID}"/>\n'
        body += text(sx(tick), 264, tick, 11, MUTED, "middle")
    # right panel: drop with 10% unseen variants (bars from 0)
    dl, dr, dmax = 640, 880, 1.5
    dx = lambda v: dl + v / dmax * (dr - dl)
    body += text(dl, 64, "drop with 10% unseen variants (points)", 12.5, MUTED)
    for tick in (0, 0.5, 1.0, 1.5):
        body += f'<line x1="{dx(tick):.1f}" y1="76" x2="{dx(tick):.1f}" y2="246" stroke="{GRID}"/>\n'
        body += text(dx(tick), 264, f"{tick:g}", 11, MUTED, "middle")

    for k, (tag, label, color) in enumerate(rows):
        y = 100 + k * 42
        values = scores(runs, "ag_news", tag)
        if not values:
            continue
        m, sd = mean_sd(values)
        body += text(30, y + 4, label, 13, INK)
        if sd:
            body += f'<line x1="{sx(m - sd):.1f}" y1="{y}" x2="{sx(m + sd):.1f}" y2="{y}" stroke="{color}" stroke-width="2"/>\n'
        body += f'<circle cx="{sx(m):.1f}" cy="{y}" r="6" fill="{color}" stroke="#ffffff" stroke-width="2"/>\n'
        body += text(sx(m) + 12, y + 4, f"{m:.1f}", 12.5, INK, weight=600)
        d, _ = mean_sd(drops(runs, "ag_news", tag))
        body += f'<rect x="{dl}" y="{y - 9}" width="{max(dx(d) - dl, 1):.1f}" height="18" rx="4" fill="{color}"/>\n'
        body += text(dx(d) + 8, y + 4, f"{d:.2f}", 12.5, INK, weight=600)
    body += text(30, 290, "MANCE: mean of 3 seeds (line = ± sd). Other rows: 1 seed.", 12, MUTED)
    return svg(width, height, body)


def main():
    runs = load()
    os.makedirs(OUT, exist_ok=True)
    for name, make in (("results.svg", results_figure), ("ablations.svg", ablations_figure)):
        with open(os.path.join(OUT, name), "w") as f:
            f.write(make(runs))
        print("wrote", os.path.join("docs", "figures", name))


if __name__ == "__main__":
    main()
