"""The pipeline: bronze -> silver (redacted) -> enrichment (cached) -> gold marts.

Enrichment results are stored in `enrich.results`, keyed by
(engine, engine version, sha256 of the redacted text). Re-running the pipeline,
adding new complaints or crashing half-way never pays twice for the same
answer; changing a prompt (a new engine version) re-enriches everything once.
Every LLM call's token usage goes to the same table, which is the cost ledger.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
from sklearn.model_selection import train_test_split

from .governance import redact

ROOT = Path(__file__).resolve().parents[2]
SAMPLE = ROOT / "data" / "complaints_2023_sample.parquet"
TAXONOMY = ROOT / "data" / "product_taxonomy.csv"
DEFAULT_DB = ROOT / "complaints.duckdb"

DDL = """
CREATE SCHEMA IF NOT EXISTS bronze;
CREATE SCHEMA IF NOT EXISTS silver;
CREATE SCHEMA IF NOT EXISTS enrich;
CREATE SCHEMA IF NOT EXISTS gold;
CREATE TABLE IF NOT EXISTS enrich.results (
    engine VARCHAR, engine_version VARCHAR, text_sha256 VARCHAR,
    product VARCHAR, confidence DOUBLE, urgency VARCHAR, summary VARCHAR,
    harm_flags VARCHAR[], input_tokens INTEGER, output_tokens INTEGER,
    cache_read_tokens INTEGER, latency_ms DOUBLE, created_at TIMESTAMP,
    PRIMARY KEY (engine, engine_version, text_sha256)
);
"""


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_bronze(con: duckdb.DuckDBPyConnection, path: Path = SAMPLE) -> int:
    con.execute(DDL)
    con.execute(f"CREATE OR REPLACE TABLE bronze.complaints AS SELECT * FROM '{path.as_posix()}'")
    return con.execute("SELECT count(*) FROM bronze.complaints").fetchone()[0]


class UnmappedProducts(ValueError):
    pass


def build_silver(con: duckdb.DuckDBPyConnection, test_size: float = 0.25, seed: int = 0) -> dict:
    """Harmonise products, redact, hash, de-duplicate, and assign a stable train/test split.

    The CFPB renamed several products during 2023 ("Credit reporting, credit repair
    services, or other..." became "Credit reporting or other..."). Labels are mapped
    to one stable taxonomy through data/product_taxonomy.csv; a product name that is
    not in the mapping stops the build rather than becoming a new class silently.
    """
    con.execute(
        f"CREATE OR REPLACE TABLE silver.product_taxonomy AS "
        f"SELECT * FROM read_csv('{TAXONOMY.as_posix()}', header = true)"
    )
    unmapped = con.execute(
        """SELECT DISTINCT b.product FROM bronze.complaints b
           LEFT JOIN silver.product_taxonomy t ON t.declared_product = b.product
           WHERE t.product IS NULL"""
    ).fetchall()
    if unmapped:
        raise UnmappedProducts(f"products missing from the taxonomy: {[u[0] for u in unmapped]}")
    df = con.execute(
        """SELECT b.* EXCLUDE (product), b.product AS declared_product, t.product
           FROM bronze.complaints b
           JOIN silver.product_taxonomy t ON t.declared_product = b.product"""
    ).df()
    red = [redact(t) for t in df["narrative"]]
    df["text"] = [r.text for r in red]
    df["pii_found"] = [json.dumps(r.found) if r.found else None for r in red]
    df["text_sha256"] = [sha(t) for t in df["text"]]
    before = len(df)
    # Copy-pasted narratives filed several times would count twice in evaluation.
    df = df.drop_duplicates("text_sha256")
    train_idx, _ = train_test_split(
        df.index, test_size=test_size, stratify=df["product"], random_state=seed
    )
    df["split"] = "test"
    df.loc[train_idx, "split"] = "train"
    df = df.drop(columns=["narrative"])  # silver never holds the unredacted text
    con.register("silver_df", df)
    con.execute("CREATE OR REPLACE TABLE silver.complaints AS SELECT * FROM silver_df")
    con.unregister("silver_df")
    return {
        "rows": len(df),
        "duplicates_removed": before - len(df),
        "with_pii_found": int(df["pii_found"].notna().sum()),
    }


def cached(con, engine, version: str, hashes: list[str]) -> set[str]:
    rows = con.execute(
        "SELECT text_sha256 FROM enrich.results WHERE engine = ? AND engine_version = ?",
        [engine, version],
    ).fetchall()
    have = {r[0] for r in rows}
    return have & set(hashes)


def enrich(con: duckdb.DuckDBPyConnection, engine, split: str | None = "test") -> dict:
    """Run `engine` on silver complaints not yet enriched by this engine version."""
    where = "WHERE split = ?" if split else ""
    df = con.execute(
        f"SELECT text_sha256, text FROM silver.complaints {where}", [split] if split else []
    ).df()
    done = cached(con, engine.name, engine.version, list(df["text_sha256"]))
    todo = df[~df["text_sha256"].isin(done)]
    if len(todo):
        results = engine.predict(list(todo["text"]))
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        con.executemany(
            "INSERT INTO enrich.results VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            [
                (engine.name, engine.version, h, r.product, r.confidence, r.urgency, r.summary,
                 r.harm_flags, r.input_tokens, r.output_tokens, r.cache_read_tokens,
                 r.latency_ms, now)
                for h, r in zip(todo["text_sha256"], results)
            ],
        )  # fmt: skip
    return {"engine": engine.name, "requested": len(df), "cached": len(done), "new": len(todo)}


def training_data(con: duckdb.DuckDBPyConnection) -> tuple[list[str], list[str]]:
    df = con.execute("SELECT text, product FROM silver.complaints WHERE split = 'train'").df()
    return list(df["text"]), list(df["product"])


def build_gold(con: duckdb.DuckDBPyConnection, engine: str) -> None:
    """BI marts on the chosen engine's enrichment (one row per complaint)."""
    con.execute(
        """
        CREATE OR REPLACE TABLE gold.complaints AS
        SELECT c.complaint_id, c.date_received, c.company, c.state, c.submitted_via,
               c.product AS product_declared, e.product AS product_routed,
               c.product != e.product AS routing_disagrees, e.confidence, e.urgency,
               e.summary, e.harm_flags, c.issue, c.company_response,
               c.timely = 'Yes' AS timely, c.pii_found IS NOT NULL AS pii_found
        FROM silver.complaints c
        JOIN enrich.results e
          ON e.text_sha256 = c.text_sha256 AND e.engine = ?
         AND e.engine_version = (SELECT max(engine_version) FROM enrich.results WHERE engine = ?)
        """,
        [engine, engine],
    )
    con.execute(
        """
        CREATE OR REPLACE TABLE gold.company_scorecard AS
        SELECT company, count(*) AS complaints,
               round(avg(timely::INT), 3) AS timely_rate,
               round(avg((company_response LIKE 'Closed with monetary relief')::INT), 3)
                   AS monetary_relief_rate,
               mode(product_routed) AS main_product
        FROM gold.complaints GROUP BY 1 HAVING count(*) >= 10 ORDER BY complaints DESC;

        CREATE OR REPLACE TABLE gold.monthly_by_product AS
        SELECT date_trunc('month', date_received) AS month, product_routed, count(*) AS complaints
        FROM gold.complaints GROUP BY ALL ORDER BY 1, 3 DESC;
        """
    )


def cost_ledger(con: duckdb.DuckDBPyConnection) -> pd.DataFrame:
    return con.execute(
        """SELECT engine, engine_version, count(*) AS calls,
                  sum(input_tokens) AS input_tokens, sum(output_tokens) AS output_tokens,
                  sum(cache_read_tokens) AS cache_read_tokens,
                  round(median(latency_ms), 1) AS p50_latency_ms
           FROM enrich.results GROUP BY 1, 2 ORDER BY 1, 2"""
    ).df()
