import re
import csv
import json
import math
import os
from collections import Counter
from typing import Dict, List, Optional, Sequence, Tuple

import torch


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def get_texts(dataset_name, datasets_map):
    """Retrieve raw texts for a single dataset by name."""
    if dataset_name not in datasets_map:
        raise ValueError(f"Dataset '{dataset_name}' not found in tag_datasets.")

    dataset = datasets_map[dataset_name]()
    data_obj = dataset[0]
    texts = getattr(data_obj, "raw_texts", None)
    if texts is None:
        raise ValueError(f"Dataset '{dataset_name}' does not provide raw_texts.")
    return texts


def _tokenize(text):
    """
    Tokenize a string into lowercase alphanumeric tokens.
    Consistent across all datasets so token frequencies are comparable.
    """
    return re.findall(r"[A-Za-z0-9]+", str(text).lower())


def _build_tfidf_matrix(
    dataset_token_counters: List[Counter],
    vocab: List[str],
    names: List[str],
) -> torch.Tensor:
    """
    Build a TF-IDF feature matrix from pre-computed per-dataset token counters.

    TF  (term frequency)   = token_count / total_tokens_in_dataset
    IDF (inverse doc freq) = log((1 + N) / (1 + df)) + 1   [smoothed]
        where N  = number of datasets
              df = number of datasets containing the token

    This weighting suppresses tokens that appear in every dataset (e.g. "the")
    and up-weights tokens that are distinctive to a particular dataset,
    giving a much better signal of dataset-level text differences than raw counts.

    Args:
        dataset_token_counters: One Counter per dataset (raw token frequencies).
        vocab: Ordered vocabulary list (output of frequency filtering).
        names: Dataset name list aligned with dataset_token_counters.

    Returns:
        feature_matrix: FloatTensor of shape [n_datasets, vocab_size], NOT normalized.
                        Callers that need cosine similarity should L2-normalize this.
    """
    token2idx = {token: idx for idx, token in enumerate(vocab)}
    n = len(names)

    # --- IDF: count how many datasets contain each vocab token ---------------
    doc_freq: Counter = Counter()
    for counter in dataset_token_counters:
        for token in counter:
            if token in token2idx:          # only track tokens in vocab
                doc_freq[token] += 1

    # Pre-compute IDF values for every vocab token.
    # Smoothed formula avoids division by zero and keeps IDF >= 1.
    idf = torch.zeros(len(vocab), dtype=torch.float32)
    for token, idx in token2idx.items():
        df = doc_freq.get(token, 0)
        idf[idx] = math.log((1 + n) / (1 + df)) + 1.0

    # --- TF × IDF ------------------------------------------------------------
    feature_matrix = torch.zeros((n, len(vocab)), dtype=torch.float32)

    for row_idx, counter in enumerate(dataset_token_counters):
        total_tokens = sum(counter.values()) or 1   # avoid division by zero
        for token, count in counter.items():
            idx = token2idx.get(token)
            if idx is not None:
                tf = count / total_tokens
                feature_matrix[row_idx, idx] = tf * idf[idx].item()

    return feature_matrix


# ---------------------------------------------------------------------------
# Public API  (drop-in replacements — same signatures as the original)
# ---------------------------------------------------------------------------

def build_text_feature_matrix(
    dataset_names=None,
    max_features=5000,
    min_token_freq=2,
    datasets_map=None,
) -> Tuple[torch.Tensor, List[str]]:
    """
    Compute pairwise TF-IDF cosine similarity matrix across graph datasets.

    Replaces the original bag-of-words approach with TF-IDF weighting so that
    tokens common to all datasets are down-weighted and dataset-specific tokens
    receive higher importance.

    Args:
        dataset_names:  Optional list of dataset names; defaults to all keys in
                        datasets_map.
        max_features:   Maximum vocabulary size (top tokens by global frequency).
        min_token_freq: Discard tokens whose global count is below this threshold.
        datasets_map:   Required mapping of {name -> dataset_factory}.

    Returns:
        matrix: FloatTensor [n, n] — pairwise cosine similarity (TF-IDF based).
        names:  Ordered dataset names aligned with matrix rows/columns.
    """
    if datasets_map is None:
        raise ValueError("datasets_map is required. Please pass tag_datasets.")

    names = list(datasets_map.keys()) if dataset_names is None else list(dataset_names)
    if not names:
        raise ValueError("dataset_names is empty.")

    # --- Collect per-dataset token counters and a global counter -------------
    dataset_token_counters: List[Counter] = []
    global_counter: Counter = Counter()

    for name in names:
        texts = get_texts(name, datasets_map)
        counter: Counter = Counter()
        for text in texts:
            counter.update(_tokenize(text))
        dataset_token_counters.append(counter)
        global_counter.update(counter)

    # --- Build vocabulary: filter by min frequency, keep top-k by frequency --
    filtered_items = [
        (token, freq)
        for token, freq in global_counter.items()
        if freq >= min_token_freq
    ]
    filtered_items.sort(key=lambda x: x[1], reverse=True)
    vocab = [token for token, _ in filtered_items[:max_features]]

    if not vocab:
        raise ValueError("No valid tokens found. Try reducing min_token_freq.")

    # --- Compute TF-IDF matrix and cosine similarity -------------------------
    feature_matrix = _build_tfidf_matrix(dataset_token_counters, vocab, names)

    norms = torch.norm(feature_matrix, p=2, dim=1, keepdim=True).clamp_min(1e-12)
    normalized = feature_matrix / norms
    matrix = normalized @ normalized.t()

    return matrix, names


def compute_text_feature_analysis(
    dataset_names: Optional[Sequence[str]] = None,
    max_features: int = 5000,
    min_token_freq: int = 2,
    datasets_map=None,
) -> Tuple[torch.Tensor, List[str], torch.Tensor, List[str], Dict[str, Dict[str, int]]]:
    """
    Full TF-IDF analysis: similarity matrix, feature matrix, vocab, and token stats.

    Compared with the original BoW version this function returns the same five
    outputs but the feature_matrix now contains TF-IDF weights instead of raw
    counts, so downstream consumers (visualisation, clustering, etc.) benefit
    from the improved weighting automatically.

    Returns:
        similarity_matrix: FloatTensor [n, n] — cosine similarity (TF-IDF).
        names:             Dataset names aligned with matrix rows/columns.
        feature_matrix:    FloatTensor [n, vocab_size] — raw (un-normalized)
                           TF-IDF weights; useful for inspecting per-token scores.
        vocab:             Vocabulary list aligned with feature_matrix columns.
        token_stats:       {dataset_name: {token: raw_count}} — original counts
                           (before TF-IDF weighting) for human-readable reporting.
    """
    if datasets_map is None:
        raise ValueError("datasets_map is required. Please pass tag_datasets.")

    names = list(datasets_map.keys()) if dataset_names is None else list(dataset_names)
    if not names:
        raise ValueError("dataset_names is empty.")

    # --- Collect per-dataset counters ----------------------------------------
    dataset_token_counters: List[Counter] = []
    global_counter: Counter = Counter()
    token_stats: Dict[str, Dict[str, int]] = {}

    for name in names:
        texts = get_texts(name, datasets_map)
        counter: Counter = Counter()
        for text in texts:
            counter.update(_tokenize(text))
        dataset_token_counters.append(counter)
        global_counter.update(counter)
        # Store raw counts for reporting (independent of weighting scheme)
        token_stats[name] = dict(counter)

    # --- Build vocabulary ----------------------------------------------------
    filtered_items = [
        (token, freq)
        for token, freq in global_counter.items()
        if freq >= min_token_freq
    ]
    filtered_items.sort(key=lambda x: x[1], reverse=True)
    vocab = [token for token, _ in filtered_items[:max_features]]

    if not vocab:
        raise ValueError("No valid tokens found. Try reducing min_token_freq.")

    # --- TF-IDF feature matrix -----------------------------------------------
    feature_matrix = _build_tfidf_matrix(dataset_token_counters, vocab, names)

    # --- Cosine similarity from L2-normalized TF-IDF vectors -----------------
    norms = torch.norm(feature_matrix, p=2, dim=1, keepdim=True).clamp_min(1e-12)
    normalized = feature_matrix / norms
    similarity_matrix = normalized @ normalized.t()

    return similarity_matrix, names, feature_matrix, vocab, token_stats


# ---------------------------------------------------------------------------
# Persistence helpers  (unchanged interface, updated docstrings)
# ---------------------------------------------------------------------------

def save_text_token_stats_json(
    names: Sequence[str],
    token_stats: Dict[str, Dict[str, int]],
    vocab: Sequence[str],
    output_json: str,
) -> None:
    """
    Save per-dataset raw token statistics to JSON.

    Note: token_stats contains raw counts (not TF-IDF weights) so the JSON
    remains human-readable and independent of the weighting scheme.
    """
    os.makedirs(os.path.dirname(output_json) or ".", exist_ok=True)
    payload = {
        "datasets": [
            {
                "name": name,
                "num_unique_tokens": len(token_stats.get(name, {})),
                "total_tokens": int(sum(token_stats.get(name, {}).values())),
            }
            for name in names
        ],
        "vocab_size": len(vocab),
        "vocab": list(vocab),
    }
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def save_text_feature_matrix_pt(
    feature_matrix: torch.Tensor,
    names: Sequence[str],
    vocab: Sequence[str],
    output_pt: str,
) -> None:
    """
    Save the TF-IDF feature matrix and metadata to a .pt file.

    The stored feature_matrix contains un-normalized TF-IDF weights
    [n_datasets, vocab_size]. Normalize before computing cosine similarity.
    """
    os.makedirs(os.path.dirname(output_pt) or ".", exist_ok=True)
    torch.save(
        {
            "feature_matrix": feature_matrix,   # TF-IDF weights, not raw counts
            "names": list(names),
            "vocab": list(vocab),
        },
        output_pt,
    )


def save_text_similarity_matrix_csv(
    similarity_matrix: torch.Tensor,
    names: Sequence[str],
    output_csv: str,
) -> None:
    """Save the n×n TF-IDF cosine similarity matrix to CSV."""
    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["dataset"] + list(names))
        for i, name in enumerate(names):
            row = similarity_matrix[i].detach().cpu().tolist()
            writer.writerow([name] + row)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_text_feature_pipeline(
    dataset_names: Optional[Sequence[str]] = None,
    max_features: int = 5000,
    min_token_freq: int = 2,
    token_stats_json: str = "z_temp/text_token_stats.json",
    feature_matrix_pt: str = "z_temp/text_feature_matrix.pt",
    similarity_matrix_csv: str = "z_temp/tfidf_similarity_matrix.csv",
    datasets_map=None,
) -> Tuple[str, str, str]:
    """
    Run the full TF-IDF text analysis pipeline and persist all stage outputs.

    Stages:
        1. Tokenize all datasets and build a shared vocabulary.
        2. Compute per-dataset TF-IDF vectors.
        3. Compute pairwise cosine similarity matrix.
        4. Save: token stats (JSON), feature matrix (.pt), similarity matrix (CSV).

    Args:
        dataset_names:        Datasets to include; None → all datasets in map.
        max_features:         Vocabulary cap (top tokens by global frequency).
        min_token_freq:       Global frequency threshold for vocabulary inclusion.
        token_stats_json:     Output path for token statistics JSON.
        feature_matrix_pt:    Output path for TF-IDF feature matrix .pt file.
        similarity_matrix_csv: Output path for cosine similarity matrix CSV.
        datasets_map:         Required {name -> dataset_factory} mapping.

    Returns:
        Tuple of the three output file paths.
    """
    similarity_matrix, names, feature_matrix, vocab, token_stats = (
        compute_text_feature_analysis(
            dataset_names=dataset_names,
            max_features=max_features,
            min_token_freq=min_token_freq,
            datasets_map=datasets_map,
        )
    )
    save_text_token_stats_json(names, token_stats, vocab, token_stats_json)
    save_text_feature_matrix_pt(feature_matrix, names, vocab, feature_matrix_pt)
    save_text_similarity_matrix_csv(similarity_matrix, names, similarity_matrix_csv)
    return token_stats_json, feature_matrix_pt, similarity_matrix_csv


if __name__ == "__main__":
    from data_provider import tag_datasets
    run_text_feature_pipeline(datasets_map=tag_datasets)
    print("TF-IDF text feature analysis completed. Results saved to z_temp/")