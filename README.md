# complaint-intelligence

[![CI](https://github.com/Adversitas/complaint-intelligence/actions/workflows/ci.yml/badge.svg)](https://github.com/Adversitas/complaint-intelligence/actions/workflows/ci.yml)

**Generative AI as a step inside a governed data pipeline. Free-text consumer complaints are:**
- harmonised and de-duplicated;
- scrubbed of personal data before any model sees them;
- enriched (routing, urgency, summary, harm flags) by interchangeable engines, from a TF-IDF baseline to local embeddings to Claude;
- evaluated against labels, then published as BI-ready marts.

Every model answer is cached by content hash and prompt version, and every token is recorded in a cost ledger.

The data is real: 7,974 complaints filed with the US Consumer Financial Protection Bureau in 2023, each with the consumer's own narrative ([CFPB Consumer Complaint Database](https://www.consumerfinance.gov/data-research/consumer-complaints/), public domain).

> **En bref (FR)** : intégration de l'IA générative dans un pipeline de données gouverné, sur 7 974 réclamations clients réelles. Le pipeline :
> - harmonise la taxonomie produits (qui a changé en 2023) et supprime les doublons ;
> - détecte et masque les données personnelles avant tout appel à un modèle ;
> - enrichit les textes (routage, urgence, résumé, signaux de préjudice) avec des moteurs interchangeables : référence TF-IDF, embeddings locaux, Claude ;
> - met les résultats en cache par empreinte du texte et version du prompt, et tient un registre des coûts ;
> - évalue chaque moteur sur des données étiquetées et publie des tables prêtes pour Power BI.

## Results

**Data quality, before any AI:**

| Step | Finding |
|---|---|
| Taxonomy | 14 declared product names collapse to 10 stable categories. The 2023 renaming would otherwise split "Credit reporting" into two classes. |
| De-duplication | **302 narratives (3.8%) were filed more than once, word for word.** Kept once, so they can't inflate the evaluation. |
| Personal data | Despite the producer's masking, **35 items of personal data** were still in published narratives: 33 URLs, 1 e-mail address and 1 phone number. They are redacted before enrichment. |

**Routing accuracy on the held-out split** (1,918 complaints, 10 products, target = product declared by the consumer):

| Engine | Accuracy | Macro-F1 | Share with confidence ≥ 0.8 | Accuracy on that share | Latency per complaint | Cost |
|---|---|---|---|---|---|---|
| **TF-IDF + logistic regression** | **77.6%** | **0.723** | 34% | **95.9%** | 0.7 ms | none |
| Local embeddings (bge-small) + logistic regression | 75.2% | 0.710 | 35% | 95.7% | 328 ms (CPU) | none |
| Claude (schema-constrained) | run with an API key: `complaints enrich --engine claude` | | | | | in `complaints costs` |

**What this shows:**

- **Cheap models handle a third of the volume very well.** About 34% of complaints get a confident prediction, and those are 96% accurate. They can be routed automatically, so the LLM or a person only sees the remaining 66%. That split is the basis for deciding where an LLM is worth its cost.
- **Embeddings did not beat TF-IDF here.** They were 2.4 points worse, and 470× slower on CPU. Product vocabulary ("escrow", "repossession", "Zelle") carries most of the signal, and a general-purpose sentence embedding blurs it.
- **The errors point at the taxonomy, not just the models.** The top confusions are payday/title loans against vehicle loans (a title loan is secured by a car), credit cards against checking accounts (debit-card disputes), and the tiny "Debt or credit management" class (43 test complaints, recall 12–28%). Some of these are consumers picking the wrong box, and the disagreement lists are where a reviewer should start.


## Pipeline

```mermaid
flowchart LR
    S[(pinned sample<br/>7,974 complaints)] --> B[bronze]
    B --> T{product taxonomy<br/>mapping}
    T -->|unmapped product| X[build stops]
    T --> R[PII scan + redaction<br/>e-mail, phone, card (Luhn),<br/>SSN, account numbers]
    R --> SV[silver<br/>redacted text, sha256,<br/>dedup, stratified split]
    SV --> E{enrichment engine}
    E --> E1[tfidf]
    E --> E2[embeddings<br/>local bge-small]
    E --> E3[Claude<br/>schema-constrained,<br/>cached prompt]
    E1 & E2 & E3 --> C[(enrich.results<br/>cache + cost ledger,<br/>key = engine, version, sha256)]
    C --> V[evaluation<br/>vs declared product]
    C --> G[gold<br/>complaints, company<br/>scorecard, monthly]
```

| Concern | How it is handled | Code |
|---|---|---|
| Changing taxonomies | The CFPB renamed several products in 2023. Labels go through a versioned mapping table ([`product_taxonomy.csv`](data/product_taxonomy.csv)), and a product missing from it **stops the build** instead of silently becoming a new class. | [`pipeline.py`](src/complaints/pipeline.py) |
| Personal data | The producer masks names and dates. The pipeline doesn't rely on that: it scans for e-mails, URLs, phone numbers, card numbers (Luhn-checked), social security numbers and long account numbers, and redacts them. Silver never stores the unredacted narrative, so no engine can receive it. | [`governance.py`](src/complaints/governance.py) |
| Paying once | Results are keyed by (engine, engine version, sha256 of the redacted text). Re-runs, crashes and new data only enrich what's missing, and a new prompt version re-enriches everything exactly once. CI checks that a second run makes zero new calls. | [`pipeline.py`](src/complaints/pipeline.py) |
| Cost visibility | Input, output and cache-read tokens and latency are recorded per call, and `complaints costs` sums them per engine and prompt version. The Claude engine caches its system prompt, which is identical for every complaint. | [`engines.py`](src/complaints/engines.py) |
| Output control | The Claude engine answers through a JSON schema: the product is an enum of the harmonised taxonomy, urgency and harm flags are enums, and the summary is short. Refusals and truncation raise errors instead of being parsed. | [`engines.py`](src/complaints/engines.py) |
| Evidence before rollout | Every engine is scored on the same held-out split, per class, with its most frequent disagreements, so the LLM has to beat the cheap baselines. | [`evaluate.py`](src/complaints/evaluate.py) |

## Run it

```bash
git clone https://github.com/Adversitas/complaint-intelligence.git
cd complaint-intelligence
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

complaints build                       # bronze -> silver
complaints enrich --engine tfidf       # run again: 0 new calls
complaints evaluate
complaints gold --engine tfidf         # gold.complaints, gold.company_scorecard, gold.monthly_by_product
pytest

pip install -e ".[embeddings]" && complaints enrich --engine embeddings
export ANTHROPIC_API_KEY=... && complaints enrich --engine claude --limit 200 && complaints costs
```

`scripts/make_sample.py` documents how the committed sample was drawn from the public CFPB snapshot.

## Limitations

- The labels are the product **declared by the consumer**, which is a noisy target. Accuracy measured against them understates a good router; the disagreement lists are where a reviewer should look first.
- The Claude engine is covered by tests with a recording fake client, but was not run on the sample here (no API key on the build machine). `complaints enrich --engine claude --limit N` produces its numbers and its cost ledger.
- The PII scanner is pattern-based. It catches structured identifiers, not every free-text detail (a street address written in words, for example). A production deployment would add a named-entity model and a policy for what may leave the platform.

## License

Code: MIT. Data: CFPB Consumer Complaint Database (public domain), 2023 complaints drawn from the snapshot mirrored at huggingface.co/datasets/BEE-spoke-data/consumer-finance-complaints (CC0). The CFPB's current exports no longer include narratives.
