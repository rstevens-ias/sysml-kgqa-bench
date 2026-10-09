# Data

All data is derived from the [TMT SysML model](https://github.com/Open-MBEE/TMT-SysML-Model) (Apache-2.0) at
commit `b2a33b71c65678498b538195dcb947a7962ca75b`. The derived data is released under
[CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) on Zenodo.

## Obtaining the data

```bash
ZENODO_DOI=<dataset DOI> bash scripts/download_data.sh   # released data from Zenodo
python -m sysml_kgqa_bench.graph.load_kuzu        # build the Kuzu database from data/graph/tmt_graph.json
```

Alternatively, every file except the LLM outputs (`benchmark/`, `results/`) can be regenerated without API
access with `scripts/01_build_graph.sh` and the first four steps of `scripts/02_generate_benchmark.sh`.

## Layout

| Path | Content | Produced by |
|---|---|---|
| `source/` | TMT model files (not redistributed) | `scripts/download_tmt.sh` |
| `processed/` | Extracted elements per kind (requirements, blocks, functions, relations, ...) and `provenance.json` | `sysml_kgqa_bench.graph.extract` |
| `graph/tmt_graph.{json,graphml}` | Property graph (37,046 nodes); schema in `docs/graph_schema.md` | `sysml_kgqa_bench.graph.build` |
| `graph/tmt_kuzu` | Kuzu database of the graph (not distributed) | `sysml_kgqa_bench.graph.load_kuzu` |
| `graph/catalogue.json` | Empirical schema: stereotypes, relations, endpoint types | `sysml_kgqa_bench.generation.catalogue` |
| `qa/candidates.jsonl` | All valid template instances (4,420) | `sysml_kgqa_bench.generation.instantiate` |
| `qa/rejections.csv` | Instances rejected by the template validity rules | `sysml_kgqa_bench.generation.instantiate` |
| `qa/candidates_sampled.jsonl` | Stratified sample (1,132) | `sysml_kgqa_bench.generation.sample` |
| `qa/*_report.md` | Sampling and leakage reports | `sysml_kgqa_bench.generation.sample`, `sysml_kgqa_bench.generation.leakage_audit` |
| `benchmark/verbalized.jsonl` | **The benchmark**: sampled instances with natural-language questions | `sysml_kgqa_bench.generation.verbalize` |
| `pcst/` | G-Retriever candidate pool and its embeddings | `sysml_kgqa_bench.baselines.build_pcst_pool`, `sysml_kgqa_bench.baselines.embed_pcst_pool` |
| `results/<baseline>/results.jsonl` | Baseline predictions and scores, as reported in the paper | `sysml_kgqa_bench.baselines.<baseline>` |

## Benchmark format

`benchmark/verbalized.jsonl` contains one JSON object per instance:

| Field | Description |
|---|---|
| `template_id` | One of the seven templates in `configs/templates/` |
| `bindings` | Graph ids bound to the template variables (e.g. `{"X": "<id>"}`) |
| `answer` | Gold answer: list of ids (T1–T3), integer (T4), paths (T5), text or `null` (T6), value (T7) |
| `gold_query` | Cypher query (or hop-wise procedure) that reproduces the answer on the Kuzu graph |
| `proof_subgraph` | Edges supporting the answer, where applicable |
| `difficulty_coords` | Template-specific difficulty attributes (closure size, depth, ...) |
| `verbalization.canonical_question` | Question rendered from the template |
| `verbalization.question` | Accepted paraphrase (or the canonical question if none was accepted) |
| `verbalization.provenance` | Models, temperatures and prompt hash used for verbalization |

Further fields name the bound entities (`seed_name`, `source_name`, `scope_name`, ...).

## Results format

Each line of `results/<baseline>/results.jsonl` holds one prediction for one question form
(`question_field` is `canonical_question` or `question`) with the raw model response, the extracted
prediction, the scores of the instance and the model provenance.
