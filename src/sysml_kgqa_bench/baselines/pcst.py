"""PCST subgraph retrieval over each instance's k-hop candidate pool (G-Retriever's retrieval step)."""

import json
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data
from torch_geometric.llm.utils.backend_utils import retrieval_via_pcst

from sysml_kgqa_bench.baselines.common import linearize_subgraph
from sysml_kgqa_bench.embeddings import get_query_embed_fn


def run_pcst(data, q_emb, textual_nodes, textual_edges, topk=3, topk_e=5, cost_e=0.5):
    """Run torch_geometric's retrieval_via_pcst in float64."""
    if data.x.dtype != torch.float64:
        data.x = data.x.to(torch.float64)
    if data.edge_attr is not None and data.edge_attr.dtype != torch.float64:
        data.edge_attr = data.edge_attr.to(torch.float64)
    q_emb = torch.as_tensor(np.array(q_emb) if not torch.is_tensor(q_emb) else q_emb, dtype=torch.float64)
    if q_emb.dim() == 1:
        q_emb = q_emb.unsqueeze(0)

    prev_default_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    try:
        return retrieval_via_pcst(data, q_emb, textual_nodes, textual_edges, topk=topk, topk_e=topk_e, cost_e=cost_e)
    finally:
        torch.set_default_dtype(prev_default_dtype)


class PCSTRetriever:
    """Retrieves a subgraph per instance from the precomputed pool and its cached embeddings."""

    def __init__(self, pool_path, embedded_path):
        with open(pool_path, encoding="utf-8") as f:
            pool = json.load(f)
        self.per_instance = pool["per_instance"]
        self.node_meta = {n["id"]: n for n in pool["global_nodes"]}

        self._edges_from = defaultdict(list)
        for e in pool["global_edges"]:
            self._edges_from[e["from"]].append(e)

        cache = np.load(embedded_path, allow_pickle=True)
        node_ids_arr = cache["node_ids"]
        node_embeddings_arr = cache["node_embeddings"]
        relation_strings_arr = cache["relation_strings"]
        relation_embeddings_arr = cache["relation_embeddings"]
        self._node_emb = {nid: node_embeddings_arr[i] for i, nid in enumerate(node_ids_arr)}
        self._rel_emb = {rel: relation_embeddings_arr[i] for i, rel in enumerate(relation_strings_arr)}
        missing_nodes = [nid for nid in self.node_meta if nid not in self._node_emb]
        if missing_nodes:
            raise RuntimeError(
                f"{len(missing_nodes)} pool nodes have no cached embedding, e.g. {missing_nodes[:5]!r}. "
                f"Re-run sysml_kgqa_bench.baselines.embed_pcst_pool."
            )

        self.question_embed_fn, self.embed_provenance = get_query_embed_fn()

    def _local_edges(self, node_id_set):
        edges = []
        for nid in node_id_set:
            for e in self._edges_from.get(nid, ()):
                if e["to"] in node_id_set:
                    edges.append(e)
        return edges

    def retrieve(self, record_key, question_text, topk=3, topk_e=5, cost_e=0.5):
        if record_key not in self.per_instance:
            raise KeyError(
                f"{record_key!r} not in the candidate pool; re-run sysml_kgqa_bench.baselines.build_pcst_pool."
            )
        local_node_ids = self.per_instance[record_key]
        node_id_set = set(local_node_ids)
        local_edges = self._local_edges(node_id_set)

        x = torch.tensor(np.stack([self._node_emb[nid] for nid in local_node_ids]))
        if local_edges:
            edge_pos = {nid: i for i, nid in enumerate(local_node_ids)}
            src = [edge_pos[e["from"]] for e in local_edges]
            dst = [edge_pos[e["to"]] for e in local_edges]
            edge_index = torch.tensor([src, dst], dtype=torch.long)
            edge_attr = torch.tensor(np.stack([self._rel_emb[e["relation"]] for e in local_edges]))
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, x.shape[1]), dtype=torch.float64)

        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        data.node_idx = np.array(local_node_ids, dtype=object)
        data.edge_idx = np.arange(len(local_edges))

        q_emb = self.question_embed_fn(question_text)

        textual_nodes = pd.DataFrame(
            {
                "node_id": range(len(local_node_ids)),
                "node_attr": [self.node_meta[nid].get("name") for nid in local_node_ids],
            }
        )
        textual_edges = pd.DataFrame(
            {
                "src": src if local_edges else [],
                "edge_attr": [e["relation"] for e in local_edges],
                "dst": dst if local_edges else [],
            }
        )

        out_data, _desc = run_pcst(data, q_emb, textual_nodes, textual_edges, topk=topk, topk_e=topk_e, cost_e=cost_e)

        selected_node_ids = list(out_data.node_idx)
        selected_edges = [local_edges[i] for i in out_data.edge_idx] if len(local_edges) else []

        nodes = [self.node_meta[nid] for nid in selected_node_ids]
        edges = [{"from": e["from"], "to": e["to"], "relation": e["relation"]} for e in selected_edges]
        return nodes, edges

    def build_context(self, record_key, question_text, topk=3, topk_e=5, cost_e=0.5):
        nodes, edges = self.retrieve(record_key, question_text, topk=topk, topk_e=topk_e, cost_e=cost_e)
        context = linearize_subgraph(nodes, edges)
        node_meta = {n["id"]: n for n in nodes}
        return context, node_meta
