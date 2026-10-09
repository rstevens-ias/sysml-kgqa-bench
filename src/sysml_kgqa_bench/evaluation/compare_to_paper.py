"""Compare the metrics computed from data/results with the numbers reported in the paper.

Each cell passes if it lies within the tolerance stored in docs/paper_reported_results.json (Clopper-Pearson
interval for 0/1 metrics, a fixed band for fractional metrics). Baselines without results are reported as
MISSING.
"""

import argparse
import json
import sys

from sysml_kgqa_bench import paths
from sysml_kgqa_bench.evaluation.analyze import to_json_summary


def compare(results_dir, reported_path, baseline_filter=None):
    with open(reported_path) as f:
        reported = json.load(f)
    fresh = to_json_summary(results_dir)

    rows = []
    for key, ref in sorted(reported.items()):
        name = key.split("/", 1)[0]
        if baseline_filter and name != baseline_filter:
            continue
        got = fresh.get(key)
        if got is None:
            rows.append((key, ref["value"], None, None, "MISSING"))
            continue
        delta = got - ref["value"]
        status = "PASS" if abs(delta) <= ref["tolerance"] else "FAIL"
        rows.append((key, ref["value"], got, delta, status))
    return rows


def print_report(rows):
    print(f"{'cell':60s} {'paper':>8s} {'this run':>10s} {'delta':>8s}  status")
    print("-" * 100)
    n_fail = 0
    for key, ref_v, got_v, delta, status in rows:
        if status == "MISSING":
            print(f"{key:60s} {ref_v:8.3f} {'--':>10s} {'--':>8s}  {status}")
            n_fail += 1
            continue
        marker = "" if status == "PASS" else "  <-- outside tolerance"
        print(f"{key:60s} {ref_v:8.3f} {got_v:10.3f} {delta:+8.3f}  {status}{marker}")
        if status == "FAIL":
            n_fail += 1
    print("-" * 100)
    print(f"{len(rows) - n_fail}/{len(rows)} cells within tolerance of the paper's reported numbers.")
    return n_fail


def write_html(rows, path):
    def esc(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;")

    trs = []
    for key, ref_v, got_v, delta, status in rows:
        color = {"PASS": "#1a7f37", "FAIL": "#cf222e", "MISSING": "#9a6700"}[status]
        got_s = f"{got_v:.3f}" if got_v is not None else "--"
        delta_s = f"{delta:+.3f}" if delta is not None else "--"
        trs.append(
            f"<tr><td>{esc(key)}</td><td>{ref_v:.3f}</td><td>{got_s}</td>"
            f"<td>{delta_s}</td><td style='color:{color};font-weight:600'>{status}</td></tr>"
        )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>compare_to_paper report</title>
<style>
body {{ font-family: -apple-system, sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ddd; padding: 4px 8px; font-size: 13px; text-align: right; }}
td:first-child, th:first-child {{ text-align: left; font-family: monospace; }}
th {{ background: #f6f8fa; text-align: right; }}
th:first-child {{ text-align: left; }}
</style></head><body>
<h2>compare_to_paper.py report</h2>
<table>
<tr><th>cell</th><th>paper</th><th>this run</th><th>delta</th><th>status</th></tr>
{"".join(trs)}
</table>
</body></html>"""
    with open(path, "w") as f:
        f.write(html)
    print(f"wrote {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default=str(paths.RESULTS))
    ap.add_argument("--reported", default=str(paths.PAPER_RESULTS))
    ap.add_argument("--baseline", default=None, choices=["oracle", "closed_book", "gretriever", "text_to_cypher"])
    ap.add_argument("--html", default=None, help="also write a static HTML report to this path")
    args = ap.parse_args()

    rows = compare(args.results_dir, args.reported, args.baseline)
    n_fail = print_report(rows)
    if args.html:
        write_html(rows, args.html)
    sys.exit(1 if n_fail else 0)
