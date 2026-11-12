"""Build the pinned sample (run once; the sample is committed).

    python scripts/make_sample.py path/to/has-text/train-00000-of-00003.parquet

Source: CFPB Consumer Complaint Database snapshot with narratives (public domain,
CC0), as mirrored at huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints
(`has-text/` shards, snapshot of early 2024). The CFPB's own export no longer
includes narratives, so a pinned snapshot is the only way to make this reproducible.

Keeps complaints received in 2023 and draws up to PER_PRODUCT per *declared*
product (stratified, seed 0). Product names are kept exactly as filed; mapping
the 2023 taxonomy change is the pipeline's job (data/product_taxonomy.csv).
"""

from __future__ import annotations

import sys
from pathlib import Path

import duckdb

OUT = Path(__file__).resolve().parents[1] / "data" / "complaints_2023_sample.parquet"
PER_PRODUCT = 600


def main(src: str) -> None:
    con = duckdb.connect()
    con.execute("SELECT setseed(0)")
    con.execute(
        f"""
        CREATE TABLE s AS
        SELECT * FROM (
            SELECT
                "Complaint ID"::BIGINT AS complaint_id,
                "Date received"::DATE AS date_received,
                "Product" AS product,
                "Sub-product" AS sub_product,
                "Issue" AS issue,
                "Consumer complaint narrative" AS narrative,
                "Company" AS company,
                "State" AS state,
                "Submitted via" AS submitted_via,
                "Company response to consumer" AS company_response,
                "Timely response?" AS timely,
                row_number() OVER (PARTITION BY "Product" ORDER BY hash("Complaint ID")) AS rn
            FROM '{Path(src).as_posix()}'
            WHERE year("Date received"::DATE) = 2023
              AND "Consumer complaint narrative" IS NOT NULL
              AND length("Consumer complaint narrative") >= 40
        )
        WHERE rn <= {PER_PRODUCT}
        ORDER BY complaint_id
        """
    )
    con.execute("ALTER TABLE s DROP COLUMN rn")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    con.execute(f"COPY s TO '{OUT.as_posix()}' (FORMAT parquet, COMPRESSION zstd)")
    print(
        con.execute("SELECT product, count(*) FROM s GROUP BY 1 ORDER BY 2 DESC").df().to_string()
    )
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main(sys.argv[1])
