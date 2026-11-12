import json
from types import SimpleNamespace

import duckdb
import pandas as pd
import pytest

from complaints import engines, evaluate, pipeline
from complaints.governance import luhn_ok, redact

# Governance -----------------------------------------------------------------------------


def test_redaction_of_personal_data():
    text = (
        "Call me at (555) 123-4567 or mail jane.doe@example.com, card 4111 1111 1111 1111, "
        "SSN 123-45-6789, account 123456789012, see https://bank.example/x. "
        "On XX/XX/XXXX I paid {$250.00} to XXXX."
    )
    r = redact(text)
    for secret in ["555", "jane.doe", "4111", "123-45-6789", "123456789012", "https://"]:
        assert secret not in r.text
    assert r.found == {"email": 1, "url": 1, "ssn": 1, "card": 1, "phone": 1, "number": 1}
    assert "[REDACTED]" in r.text and "XXXX" not in r.text


def test_card_numbers_need_a_valid_checksum():
    assert luhn_ok("4111111111111111") and not luhn_ok("4111111111111112")
    # 16 digits failing Luhn are not a card, but still a long reference number.
    assert redact("ref 4111111111111112").found == {"number": 1}


def test_clean_text_is_untouched():
    r = redact("My   loan servicer charged a late fee twice.")
    assert r.text == "My loan servicer charged a late fee twice." and not r.leaked


# Pipeline on a small synthetic sample ------------------------------------------------------


@pytest.fixture
def con(tmp_path):
    rows = []
    topics = {
        "Mortgage": "my mortgage servicer added escrow fees and threatened foreclosure on the house",
        "Debt collection": "a debt collector keeps calling about a debt I do not owe",
        "Student loan": "my student loan servicer lost my income driven repayment application",
    }
    declared = {"Mortgage": "Mortgage", "Debt collection": "Debt collection",
                "Student loan": "Student loan"}  # fmt: skip
    i = 0
    for product, text in topics.items():
        for k in range(40):
            rows.append({
                "complaint_id": i, "date_received": pd.Timestamp("2023-03-01"),
                "product": declared[product], "sub_product": None, "issue": "x",
                "narrative": f"{text} case {k} on XX/XX/XXXX", "company": f"Co {k % 3}",
                "state": "CA", "submitted_via": "Web", "company_response": "Closed with explanation",
                "timely": "Yes",
            })  # fmt: skip
            i += 1
    rows.append({**rows[0], "complaint_id": 999})  # an exact duplicate narrative
    path = tmp_path / "sample.parquet"
    pd.DataFrame(rows).to_parquet(path)
    c = duckdb.connect(str(tmp_path / "t.duckdb"))
    pipeline.load_bronze(c, path)
    return c


def test_silver_dedups_redacts_and_splits(con):
    stats = pipeline.build_silver(con)
    assert stats["duplicates_removed"] == 1 and stats["rows"] == 120
    cols = [r[0] for r in con.execute("DESCRIBE silver.complaints").fetchall()]
    assert "narrative" not in cols  # unredacted text never reaches silver
    splits = dict(
        con.execute("SELECT split, count(*) FROM silver.complaints GROUP BY 1").fetchall()
    )
    assert splits == {"train": 90, "test": 30}


def test_unmapped_product_stops_the_build(con):
    con.execute("UPDATE bronze.complaints SET product = 'Brand new product' WHERE complaint_id = 1")
    with pytest.raises(pipeline.UnmappedProducts, match="Brand new product"):
        pipeline.build_silver(con)


def test_enrichment_is_cached_and_versioned(con):
    pipeline.build_silver(con)
    engine = engines.tfidf_engine().fit(*pipeline.training_data(con))
    first = pipeline.enrich(con, engine)
    again = pipeline.enrich(con, engine)
    assert first["new"] == 30 and again == {**again, "new": 0, "cached": 30}
    engine.version = "2"  # a new model or prompt version re-enriches everything once
    assert pipeline.enrich(con, engine)["new"] == 30
    report = evaluate.evaluate(con, "tfidf", "2")
    assert report["n"] == 30 and report["accuracy"] == 1.0


def test_gold_marts(con):
    pipeline.build_silver(con)
    engine = engines.tfidf_engine().fit(*pipeline.training_data(con))
    pipeline.enrich(con, engine)
    pipeline.build_gold(con, "tfidf")
    n, disagreements = con.execute(
        "SELECT count(*), sum(routing_disagrees::INT) FROM gold.complaints"
    ).fetchone()
    assert n == 30 and disagreements == 0
    assert con.execute("SELECT count(*) FROM gold.company_scorecard").fetchone()[0] >= 1


# Claude engine (fake client) ----------------------------------------------------------------


class FakeMessages:
    def __init__(self):
        self.calls = []

    def create(self, **kw):
        self.calls.append(kw)
        out = {"product": "Mortgage", "urgency": "high", "summary": "Escrow dispute.",
               "harm_flags": ["foreclosure_or_eviction"]}  # fmt: skip
        return SimpleNamespace(
            stop_reason="end_turn",
            content=[SimpleNamespace(type="text", text=json.dumps(out))],
            usage=SimpleNamespace(input_tokens=900, output_tokens=40, cache_read_input_tokens=800),
        )


def test_claude_engine_request_shape_and_ledger(con):
    pipeline.build_silver(con)
    fake = FakeMessages()
    client = SimpleNamespace(beta=SimpleNamespace(messages=fake))
    engine = engines.ClaudeEngine(["Debt collection", "Mortgage", "Student loan"], client=client)
    assert pipeline.enrich(con, engine)["new"] == 30
    call = fake.calls[0]
    assert call["system"][0]["cache_control"] == {"type": "ephemeral"}
    enum = call["output_config"]["format"]["schema"]["properties"]["product"]["enum"]
    assert enum == ["Debt collection", "Mortgage", "Student loan"]
    assert "XX/XX/XXXX" not in call["messages"][0]["content"]  # only redacted text is sent
    ledger = pipeline.cost_ledger(con).set_index("engine").loc["claude"]
    assert ledger["calls"] == 30 and ledger["input_tokens"] == 27_000
    assert ledger["cache_read_tokens"] == 24_000


def test_claude_engine_raises_on_refusal():
    fake = FakeMessages()
    fake.create = lambda **kw: SimpleNamespace(stop_reason="refusal", content=[], usage=None)
    engine = engines.ClaudeEngine(
        ["Mortgage"], client=SimpleNamespace(beta=SimpleNamespace(messages=fake))
    )
    with pytest.raises(RuntimeError, match="refusal"):
        engine.predict(["text"])
