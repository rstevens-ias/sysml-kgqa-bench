"""Default file locations, relative to the repository root."""

import os
from pathlib import Path

ROOT = Path(os.environ.get("SYSML_KGQA_BENCH_ROOT", Path(__file__).resolve().parents[2]))

CONFIGS = ROOT / "configs"
MODEL_CONFIG = CONFIGS / "model_config.yaml"
TEMPLATES = CONFIGS / "templates"

DOCS = ROOT / "docs"
GRAPH_SCHEMA_DOC = DOCS / "graph_schema.md"
PAPER_RESULTS = DOCS / "paper_reported_results.json"

DATA = ROOT / "data"
SOURCE = DATA / "source"
MTIP_XML = SOURCE / "TMT_mtip.xml"
MDZIP_MODEL = SOURCE / "com.nomagic.magicdraw.uml_model.model"
PROCESSED = DATA / "processed"
GRAPH = DATA / "graph"
GRAPH_JSON = GRAPH / "tmt_graph.json"
KUZU_DB = GRAPH / "tmt_kuzu"
KUZU_CSV = GRAPH / "kuzu_csv"
CATALOGUE = GRAPH / "catalogue.json"
QA = DATA / "qa"
CANDIDATES = QA / "candidates.jsonl"
CANDIDATES_SAMPLED = QA / "candidates_sampled.jsonl"
BENCHMARK = DATA / "benchmark"
VERBALIZED = BENCHMARK / "verbalized.jsonl"
PCST_POOL = DATA / "pcst" / "pool.json"
PCST_EMBEDDINGS = DATA / "pcst" / "pool_embeddings.npz"
RESULTS = DATA / "results"
