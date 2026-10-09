<a id="readme-top"></a>

<div align="center">

  <a href="https://www.python.org/">
    <img alt="Python 3.12" src="https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python">
  </a>
  <a href="https://pytorch.org/">
    <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-_-orange?style=for-the-badge&logo=pytorch">
  </a>
  <a href="https://pytorch-geometric.readthedocs.io/">
    <img alt="PyG" src="https://img.shields.io/badge/PyG-_-lightblue?style=for-the-badge&logo=pyg">
  </a>
  <a href="https://docs.docker.com/">
    <img alt="Docker" src="https://img.shields.io/badge/Docker-_-blue?style=for-the-badge&logo=docker">
  </a>
  <a href="LICENSE">
    <img alt="License: MIT" src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge">
  </a>

</div>

<br />
<div align="center">
  <h3 align="center">Same Graph, Opposite Criteria</h3>
  <p align="center">
    Per-Class Correctness Guarantees for Closed-World KGQA
    <br />
    Official code and data for the extended abstract at the Learning on Graphs Conference (LoG) 2026
    <br />
    <br />
    <a href="https://github.com/rstevens-ias/sysml-kgqa-bench/issues">Report a bug</a>
  </p>
</div>

<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#about-the-project">About The Project</a></li>
    <li><a href="#results">Results</a></li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
        <li><a href="#data">Data</a></li>
      </ul>
    </li>
    <li><a href="#reproducing-the-paper">Reproducing the Paper</a></li>
    <li><a href="#repository-structure">Repository Structure</a></li>
    <li><a href="#citation">Citation</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project

This repository contains a closed-world knowledge graph question answering (KGQA) benchmark built from the
SysML model of the Thirty Meter Telescope (TMT). Questions are generated from seven templates over one
knowledge graph (37,046 nodes). Each template class comes with its own correctness guarantee and metric,
so the same graph is evaluated under different criteria:

| Template | Question type | Guarantee | Metric |
|---|---|---|---|
| T1 `impact_analysis.requirement_closure` | Which elements are affected if a requirement changes? | completeness | completeness / faithfulness |
| T2 `negation.requirement_unrefined` | Which requirements in a cluster are not refined? | exactness | exact set match |
| T3 `impact_analysis.structural_closure` | Which elements are affected if a block or property changes? | completeness | completeness / faithfulness |
| T4 `aggregation.requirement_count` | How many active requirements are in a scope? | exactness | exact match |
| T5 `tracing.requirement_derivation` | How is one requirement derived from another? | faithfulness | edge existence / endpoint connection |
| T6 `rationale_abstention` | What is the rationale of a requirement? | abstention | abstention accuracy / LASM accuracy |
| T7 `lookup.value_property_attribute` | What is the default value of a property? | exactness | exact match |

The benchmark contains 1,132 instances. Each instance has a template-generated canonical question and an LLM
paraphrase (the canonical question where no paraphrase passed the filters). Four baselines are
evaluated: closed-book, G-Retriever (retrieval only), text-to-Cypher, and an oracle that receives the gold
proof subgraph.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- RESULTS -->
## Results

Primary metric per template and baseline (LLM: `qwen-3-5-397b-a17b`, averaged over both question forms):

| Template | Metric | Closed-book | G-Retriever | Text-to-Cypher | Oracle |
|---|---|---:|---:|---:|---:|
| T1 Impact analysis (requirements) | completeness | 0.000 | 0.727 | 0.807 | 1.000 |
| T2 Negation | exact match | 0.000 | 0.240 | 0.312 | 0.594 |
| T3 Impact analysis (structural) | completeness | 0.001 | 0.228 | 0.122 | 0.786 |
| T4 Aggregation | exact match | 0.000 | 0.159 | 0.184 | 1.000 |
| T5 Tracing | endpoint connection | 0.000 | 0.657 | 0.576 | 0.998 |
| T6 Rationale / abstention | accuracy | 0.170 | 0.940 | 1.000 | 0.987 |
| T7 Lookup | exact match | 0.000 | 0.905 | 0.847 | 0.976 |

All reported numbers, including secondary metrics, are stored in
[`docs/paper_reported_results.json`](docs/paper_reported_results.json) and can be recomputed from the released
predictions with `bash scripts/05_evaluate.sh`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- GETTING STARTED -->
## Getting Started

### Prerequisites

* [uv](https://docs.astral.sh/uv/) and Python 3.12, or Docker
* Bash, `curl` and `unzip` for the download scripts
* Optional: an NVIDIA GPU (the embedding model also runs on CPU)
* API access only for regenerating LLM outputs: the OpenAI API for verbalization, and an OpenAI-compatible
  endpoint (e.g. [vLLM](https://docs.vllm.ai/)) serving the baseline model

### Installation

```sh
git clone https://github.com/rstevens-ias/sysml-kgqa-bench.git
cd sysml-kgqa-bench
```

With uv:
```sh
uv sync --extra cpu          # or: uv sync --extra cu128
source .venv/bin/activate
```

With Docker:
```sh
docker compose up -d --build
docker compose exec sysml-kgqa-bench bash
```

For runs that call an LLM, copy `.env.example` to `.env` and fill in the credentials.

### Data

The benchmark, the intermediate data and all baseline predictions are archived on Zenodo:
```sh
ZENODO_DOI=<dataset DOI> bash scripts/download_data.sh
python -m sysml_kgqa_bench.graph.load_kuzu    # builds the Kuzu graph database
```
See [`data/README.md`](data/README.md) for the layout and the record format.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- REPRODUCING -->
## Reproducing the Paper

The pipeline consists of five bash scripts (e.g. `bash scripts/05_evaluate.sh`). Each step can be run on its own using
the released data from the previous steps.

| Script | Step | LLM calls |
|---|---|---|
| `scripts/01_build_graph.sh` | Download the TMT model, extract it, build the graph and the Kuzu database | no |
| `scripts/02_generate_benchmark.sh` | Instantiate the templates, sample, audit leakage and verbalize the questions | verbalization only |
| `scripts/03_run_checks.sh` | Tests, trivial floors and dry runs of the baselines | no |
| `scripts/04_run_baselines.sh [name\|all]` | Run the baselines on the full benchmark | yes |
| `scripts/05_evaluate.sh` | Print the result tables and compare them with the paper | no |

Everything up to the verbalization step is deterministic and reproduces the released files exactly. LLM
outputs (paraphrases and baseline answers) can vary between runs; `scripts/05_evaluate.sh` reports for every
reported number whether a new run lies within its tolerance (Clopper-Pearson interval or a fixed band).

Individual modules can also be run directly, e.g. `python -m sysml_kgqa_bench.baselines.oracle --help`.

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- STRUCTURE -->
## Repository Structure

```
configs/
  model_config.yaml        model-specific stereotypes and relation directions
  templates/               the seven question templates
data/                      downloaded or generated data (see data/README.md)
docs/
  graph_schema.md          node and edge types of the graph
  paper_reported_results.json
scripts/                   pipeline entry points (01-05) and download scripts
src/sysml_kgqa_bench/
  graph/                   extraction, graph construction, Kuzu loading
  generation/              catalogue, template instantiation, sampling, leakage audit, verbalization
  baselines/               closed-book, oracle, G-Retriever, text-to-Cypher, trivial floors
  evaluation/              answer extraction, metrics, result tables
  embeddings.py, llm.py    embedding model and LLM clients
tests/                     metric, extraction and identity tests
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CITATION -->
## Citation

```bibtex
@inproceedings{stevens2026samegraph,
  title     = {Same Graph, Opposite Criteria: Per-Class Correctness Guarantees for Closed-World {KGQA}},
  author    = {Stevens, Robin and Baum, Sebastian and Jazdi, Nasser and Weyrich, Michael},
  booktitle = {Learning on Graphs Conference (LoG), Extended Abstract Track},
  year      = {2026}
}
```

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- LICENSE -->
## License

The code is released under the MIT License, see [`LICENSE`](LICENSE). The data on Zenodo is released under
CC BY 4.0 and is derived from the [TMT SysML model](https://github.com/Open-MBEE/TMT-SysML-Model) (Apache-2.0).

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- ACKNOWLEDGMENTS -->
## Acknowledgments

* [Open-MBEE TMT SysML model](https://github.com/Open-MBEE/TMT-SysML-Model)
* [G-Retriever](https://github.com/XiaoxinHe/G-Retriever) and its implementation in [PyG](https://pytorch-geometric.readthedocs.io/)
* [Kuzu](https://github.com/kuzudb/kuzu)
* [Best-README-Template](https://github.com/othneildrew/Best-README-Template)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
