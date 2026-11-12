"""Command-line interface.

complaints build                         bronze -> silver (harmonised, redacted, split)
complaints enrich --engine tfidf         enrich the test split (cached, idempotent)
complaints enrich --engine claude        same with Claude (needs ANTHROPIC_API_KEY)
complaints evaluate                      compare every engine that has results
complaints gold --engine tfidf           build BI marts from one engine's results
complaints costs                         token ledger per engine and prompt version
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import duckdb

from . import evaluate, pipeline


def make_engine(name: str, con: duckdb.DuckDBPyConnection):
    from . import engines

    if name == "claude":
        products = [r[0] for r in con.execute(
            "SELECT DISTINCT product FROM silver.complaints").fetchall()]  # fmt: skip
        return engines.ClaudeEngine(products)
    engine = engines.tfidf_engine() if name == "tfidf" else engines.embedding_engine()
    return engine.fit(*pipeline.training_data(con))


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(
        prog="complaints", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--db", default=str(pipeline.DEFAULT_DB))
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("build")
    sp = sub.add_parser("enrich")
    sp.add_argument("--engine", choices=["tfidf", "embeddings", "claude"], required=True)
    sp.add_argument("--split", default="test", help="test (default), train, or all")
    sp.add_argument("--limit", type=int, help="enrich at most N complaints (cost control)")
    sp = sub.add_parser("evaluate")
    sp.add_argument("--output", type=Path)
    sp = sub.add_parser("gold")
    sp.add_argument("--engine", default="tfidf")
    sub.add_parser("costs")
    args = p.parse_args(argv)

    con = duckdb.connect(args.db)
    con.execute(pipeline.DDL)
    if args.command == "build":
        print("bronze rows:", pipeline.load_bronze(con))
        print("silver:", json.dumps(pipeline.build_silver(con)))
    elif args.command == "enrich":
        engine = make_engine(args.engine, con)
        if args.limit:
            engine = _Limited(engine, args.limit)
        split = None if args.split == "all" else args.split
        print(json.dumps(pipeline.enrich(con, engine, split)))
    elif args.command == "evaluate":
        rows = con.execute(
            "SELECT engine, max(engine_version) FROM enrich.results GROUP BY 1 ORDER BY 1"
        ).fetchall()
        report = {}
        for engine, version in rows:
            report[engine] = evaluate.evaluate(con, engine, version)
            print(json.dumps(report[engine]))
            print(evaluate.per_class(con, engine, version).to_string(index=False))
            print("most frequent disagreements:")
            print(evaluate.top_confusions(con, engine, version).to_string(index=False), "\n")
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2))
    elif args.command == "gold":
        pipeline.build_gold(con, args.engine)
        print(con.execute("SELECT count(*) FROM gold.complaints").fetchone()[0], "rows in gold")
    elif args.command == "costs":
        print(pipeline.cost_ledger(con).to_string(index=False))
    con.close()


class _Limited:
    """Wraps an engine so that a run enriches at most `limit` new complaints."""

    def __init__(self, engine, limit: int) -> None:
        self.engine, self.limit = engine, limit
        self.name, self.version = engine.name, engine.version

    def predict(self, texts):
        return self.engine.predict(texts[: self.limit])


if __name__ == "__main__":
    main()
