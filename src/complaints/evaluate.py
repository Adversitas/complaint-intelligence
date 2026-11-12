"""Evaluation of enrichment engines against the declared product on the test split.

The CFPB `product` field is chosen by the consumer when filing, so it is a noisy
label: some disagreements are the engine being right and the consumer picking
the wrong box. The report therefore also lists the most frequent disagreements,
which is where a human should look before trusting any engine's accuracy.
"""

from __future__ import annotations

import duckdb
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score


def evaluate(con: duckdb.DuckDBPyConnection, engine: str, version: str) -> dict:
    df = con.execute(
        """SELECT c.product AS gold, e.product AS pred, e.confidence
           FROM silver.complaints c JOIN enrich.results e ON e.text_sha256 = c.text_sha256
           WHERE c.split = 'test' AND e.engine = ? AND e.engine_version = ?""",
        [engine, version],
    ).df()
    if df.empty:
        return {"engine": engine, "n": 0}
    out = {
        "engine": engine,
        "version": version,
        "n": len(df),
        "accuracy": round(float(accuracy_score(df["gold"], df["pred"])), 4),
        "macro_f1": round(float(f1_score(df["gold"], df["pred"], average="macro")), 4),
    }
    if df["confidence"].notna().all():
        confident = df[df["confidence"] >= 0.8]
        out["share_confident_0_8"] = round(len(confident) / len(df), 4)
        out["accuracy_when_confident"] = round(
            float((confident["gold"] == confident["pred"]).mean()), 4
        )
    return out


def per_class(con: duckdb.DuckDBPyConnection, engine: str, version: str) -> pd.DataFrame:
    return con.execute(
        """SELECT c.product, count(*) AS n,
                  round(avg((c.product = e.product)::INT), 3) AS recall
           FROM silver.complaints c JOIN enrich.results e ON e.text_sha256 = c.text_sha256
           WHERE c.split = 'test' AND e.engine = ? AND e.engine_version = ?
           GROUP BY 1 ORDER BY recall""",
        [engine, version],
    ).df()


def top_confusions(con: duckdb.DuckDBPyConnection, engine: str, version: str, k: int = 8):
    return con.execute(
        """SELECT c.product AS declared, e.product AS predicted, count(*) AS n
           FROM silver.complaints c JOIN enrich.results e ON e.text_sha256 = c.text_sha256
           WHERE c.split = 'test' AND e.engine = ? AND e.engine_version = ?
             AND c.product != e.product
           GROUP BY 1, 2 ORDER BY n DESC LIMIT ?""",
        [engine, version, k],
    ).df()
