"""Per-month top-25 predicted-shortage dashboard, rendered to GitHub Markdown.

Trains the production CatBoost + within-month-rank blend (same wiring as
`shortage_forecast.operational`), then for every month in the test split
writes the top 25 predicted DINs to a Markdown page that GitHub will render
inline. The top 10 are emphasised in their own table inside a GitHub
`[!IMPORTANT]` callout — the project headlines both P@10 and P@25, and the
visual split makes the two cohorts easy to scan.

Each row carries:
  - Rank, predicted-risk percentile within the month
  - DIN, brand name, ingredient(s), manufacturer
  - A ✅ / ⬜ flag for whether a shortage actually started within 90 days

The output lives at `dashboard/` in the repo root:
  dashboard/
    README.md       # index — links to every month + per-month P@10 / P@25
    YYYY-MM.md      # one page per test month, with prev / next nav

Usage:
    python -m shortage_forecast.dashboard
"""

from __future__ import annotations

from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

from shortage_forecast.baseline import build_features, predict_proba, train_model
from shortage_forecast.config import PROJECT_ROOT, TARGET, get_db_path
from shortage_forecast.data_loader import load_splits
from shortage_forecast.operational import (
    score_blended,
    score_heuristic_single,
)

DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
TOP_K_HIGHLIGHT = 10
TOP_K_TOTAL = 25
HIT_FLAG = "✅"
MISS_FLAG = "⬜"


# ---------------------------------------------------------------------------
# Drug descriptions
# ---------------------------------------------------------------------------

def load_drug_descriptions(db_path: Path) -> pd.DataFrame:
    """One row per DIN with brand / ingredient / manufacturer for display."""
    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute(
            """
            SELECT
                din,
                brand_name,
                descriptor,
                ingredients_list,
                company_name
            FROM main_intermediate.dim_drug_by_din
            """
        ).fetchdf()
    finally:
        con.close()
    return df


def _clean_cell(value: object) -> str:
    """Markdown-safe single-line cell value, with `|` escaped."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "—"
    text = str(value).strip()
    if not text:
        return "—"
    text = text.replace("|", "\\|").replace("\n", " ")
    return text


def _format_brand(row: pd.Series) -> str:
    brand = _clean_cell(row.get("brand_name"))
    descriptor = _clean_cell(row.get("descriptor"))
    if descriptor != "—" and descriptor.lower() not in brand.lower():
        return f"{brand} ({descriptor})"
    return brand


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_test_set(
    train: pd.DataFrame,
    val: pd.DataFrame,
    test: pd.DataFrame,
) -> np.ndarray:
    """Return blended within-month-rank scores on the test split."""
    X_train, y_train = build_features(train)
    X_val,   y_val   = build_features(val)
    models = train_model(X_train, y_train, X_val, y_val)
    gbm_scores = predict_proba(models["catboost"], test)
    heur_scores = score_heuristic_single(test)
    return score_blended(test, gbm_scores, heur_scores)


# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------

TABLE_HEADER = (
    "| Rank | DIN | Brand (descriptor) | Ingredient(s) | Manufacturer | Shortage in 90d? |\n"
    "|-----:|-----|--------------------|---------------|--------------|:----------------:|"
)


def _render_rows(df: pd.DataFrame, start_rank: int) -> str:
    lines: list[str] = []
    for offset, (_, row) in enumerate(df.iterrows()):
        rank = start_rank + offset
        flag = HIT_FLAG if int(row[TARGET]) == 1 else MISS_FLAG
        lines.append(
            "| {rank} | `{din}` | {brand} | {ingr} | {mfr} | {flag} |".format(
                rank=rank,
                din=_clean_cell(row["din"]),
                brand=_format_brand(row),
                ingr=_clean_cell(row.get("ingredients_list")),
                mfr=_clean_cell(row.get("company_name")),
                flag=flag,
            )
        )
    return "\n".join(lines)


def _month_filename(month: pd.Timestamp) -> str:
    return f"{month.strftime('%Y-%m')}.md"


def _month_label(month: pd.Timestamp) -> str:
    return month.strftime("%B %Y")


def render_month_page(
    month: pd.Timestamp,
    ranked: pd.DataFrame,
    prev_month: pd.Timestamp | None,
    next_month: pd.Timestamp | None,
) -> str:
    """Render the per-month page. `ranked` is already sorted descending by score
    and trimmed to TOP_K_TOTAL rows."""
    top = ranked.iloc[:TOP_K_HIGHLIGHT]
    rest = ranked.iloc[TOP_K_HIGHLIGHT:TOP_K_TOTAL]

    p10 = top[TARGET].mean() if len(top) else float("nan")
    p25 = ranked[TARGET].mean() if len(ranked) else float("nan")
    hits10 = int(top[TARGET].sum())
    hits25 = int(ranked[TARGET].sum())

    nav_parts: list[str] = []
    if prev_month is not None:
        nav_parts.append(f"[← {_month_label(prev_month)}]({_month_filename(prev_month)})")
    nav_parts.append("[Index](README.md)")
    if next_month is not None:
        nav_parts.append(f"[{_month_label(next_month)} →]({_month_filename(next_month)})")
    nav = " · ".join(nav_parts)

    lines = [
        f"# Top {TOP_K_TOTAL} predicted shortages — {_month_label(month)}",
        "",
        nav,
        "",
        "Predictions from the production CatBoost + within-month-rank blend, ",
        f"scored on `{month.date().isoformat()}`. The flag in the last column shows whether the ",
        "drug actually entered shortage within the next 90 days.",
        "",
        "> [!IMPORTANT]",
        f"> **P@10:** {p10:.2f} ({hits10}/{TOP_K_HIGHLIGHT})  ·  "
        f"**P@25:** {p25:.2f} ({hits25}/{TOP_K_TOTAL})",
        "",
        "## 🎯 Top 10 — highest predicted risk",
        "",
        TABLE_HEADER,
    ]
    top_rows = _render_rows(top, start_rank=1)
    if top_rows:
        lines.append(top_rows)
    lines += [
        "",
        "<br>",
        "",
        f"## Ranks 11–{TOP_K_TOTAL}",
        "",
        TABLE_HEADER,
    ]
    rest_rows = _render_rows(rest, start_rank=TOP_K_HIGHLIGHT + 1)
    if rest_rows:
        lines.append(rest_rows)
    lines += [
        "",
        "---",
        nav,
        "",
    ]
    return "\n".join(lines)


def render_index(monthly_summaries: list[dict]) -> str:
    """Index page listing every month with its P@10 / P@25."""
    lines = [
        "# Drug shortage forecast — monthly top-25 dashboard",
        "",
        "Per-month top-25 ranked predictions on the 10-month test split ",
        "(2025-04 … 2026-01). Each page shows the DINs the production ",
        "CatBoost + heuristic blend ranked highest for that month, with a ✅ / ⬜ ",
        "flag for whether a Health Canada shortage actually started within 90 days.",
        "",
        "The top 10 and ranks 11–25 are split into separate tables on every ",
        "page because the project reports both **P@10** and **P@25** as ",
        "headline metrics.",
        "",
        "## Months",
        "",
        "| Month | P@10 | P@25 | Hits @10 | Hits @25 |",
        "|-------|-----:|-----:|:--------:|:--------:|",
    ]
    for s in monthly_summaries:
        lines.append(
            "| [{label}]({file}) | {p10:.2f} | {p25:.2f} | {h10}/{k10} | {h25}/{k25} |".format(
                label=s["label"],
                file=s["file"],
                p10=s["p10"],
                p25=s["p25"],
                h10=s["hits10"],
                k10=TOP_K_HIGHLIGHT,
                h25=s["hits25"],
                k25=TOP_K_TOTAL,
            )
        )

    overall_p10 = float(np.mean([s["p10"] for s in monthly_summaries]))
    overall_p25 = float(np.mean([s["p25"] for s in monthly_summaries]))
    lines += [
        "",
        f"**Mean per-month P@10:** {overall_p10:.3f}  ·  "
        f"**Mean per-month P@25:** {overall_p25:.3f}",
        "",
        "Generated by `python -m shortage_forecast.dashboard`.",
        "",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def main() -> None:
    DASHBOARD_DIR.mkdir(parents=True, exist_ok=True)

    train, val, test = load_splits()
    print(
        f"Loaded test split: {len(test):,} rows across "
        f"{test['observation_date'].nunique()} months"
    )

    scores = score_test_set(train, val, test)
    test = test.assign(_score=scores)

    descriptions = load_drug_descriptions(get_db_path())
    print(f"Loaded {len(descriptions):,} drug descriptions from dim_drug_by_din.")

    months = sorted(test["observation_date"].unique())
    monthly_summaries: list[dict] = []

    for i, month in enumerate(months):
        month_ts = pd.Timestamp(month)
        month_df = test.loc[test["observation_date"] == month].copy()
        ranked = (
            month_df.sort_values("_score", ascending=False)
            .head(TOP_K_TOTAL)
            .merge(descriptions, on="din", how="left")
            .reset_index(drop=True)
        )
        prev_month = pd.Timestamp(months[i - 1]) if i > 0 else None
        next_month = pd.Timestamp(months[i + 1]) if i + 1 < len(months) else None

        page = render_month_page(month_ts, ranked, prev_month, next_month)
        out_path = DASHBOARD_DIR / _month_filename(month_ts)
        out_path.write_text(page, encoding="utf-8")
        print(f"  wrote {out_path.relative_to(PROJECT_ROOT)}")

        top = ranked.iloc[:TOP_K_HIGHLIGHT]
        monthly_summaries.append({
            "label":  _month_label(month_ts),
            "file":   _month_filename(month_ts),
            "p10":    float(top[TARGET].mean()) if len(top) else float("nan"),
            "p25":    float(ranked[TARGET].mean()) if len(ranked) else float("nan"),
            "hits10": int(top[TARGET].sum()),
            "hits25": int(ranked[TARGET].sum()),
        })

    index_path = DASHBOARD_DIR / "README.md"
    index_path.write_text(render_index(monthly_summaries), encoding="utf-8")
    print(f"  wrote {index_path.relative_to(PROJECT_ROOT)}")
    print(f"\nDashboard ready at {DASHBOARD_DIR.resolve()}")


__all__ = [
    "DASHBOARD_DIR",
    "TOP_K_HIGHLIGHT",
    "TOP_K_TOTAL",
    "load_drug_descriptions",
    "main",
    "render_index",
    "render_month_page",
    "score_test_set",
]


if __name__ == "__main__":
    main()
