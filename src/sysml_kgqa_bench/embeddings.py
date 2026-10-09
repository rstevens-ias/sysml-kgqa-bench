"""Local sentence embeddings (Qwen3-Embedding-0.6B) used by LASM scoring and PCST retrieval."""

import functools
import os
import time

MODEL = "Qwen/Qwen3-Embedding-0.6B"


class LocalEmbeddingClient:
    def __init__(self, model=MODEL, device=None):
        from sentence_transformers import SentenceTransformer

        self.model_name = model
        self._model = SentenceTransformer(model, device=device)

    def embed(self, texts, prompt_name=None):
        return self._model.encode(list(texts), prompt_name=prompt_name, convert_to_numpy=True).tolist()


@functools.lru_cache(maxsize=1)
def get_client():
    """Load the embedding model once, on GPU if available, otherwise on CPU."""
    import torch

    if torch.cuda.is_available():
        try:
            t0 = time.time()
            client = LocalEmbeddingClient(device="cuda")
            client.embed(["connectivity check"])
            print(f"Embeddings: {MODEL} on GPU, loaded in {time.time() - t0:.1f}s.")
            return client, {"embedding_backend": "local", "embedding_model": MODEL, "device": "cuda"}
        except Exception as e:
            print(f"WARNING: {MODEL} failed on GPU ({type(e).__name__}: {e}), falling back to CPU.", flush=True)

    torch.set_num_threads(os.cpu_count() or 8)
    t0 = time.time()
    client = LocalEmbeddingClient(device="cpu")
    client.embed(["connectivity check"])
    print(f"Embeddings: {MODEL} on CPU, loaded in {time.time() - t0:.1f}s.")
    return client, {"embedding_backend": "local", "embedding_model": MODEL, "device": "cpu"}


@functools.lru_cache(maxsize=4096)
def _embed_cached(text, prompt_name=None):
    client, _ = get_client()
    return tuple(client.embed([text], prompt_name=prompt_name)[0])


def get_default_embed_fn():
    """Return (embed_fn, provenance) for document-side embeddings."""
    _, provenance = get_client()
    return (lambda text: list(_embed_cached(text))), provenance


def get_query_embed_fn():
    """Return (embed_fn, provenance) for query-side embeddings (uses the model's query prompt)."""
    _, provenance = get_client()
    return (lambda text: list(_embed_cached(text, prompt_name="query"))), provenance
