"""
eval.py — Evaluate a trained embedding model.

Launch:
    python eval.py --results_dir ./results_20240101_120000 --model ./checkpoints/model_9.pth

The model .pth can also be dropped directly in results_dir and will be auto-detected.

Computes (from PLAN.md):
  - kNN classifier          : train embeddings as gallery, test as query
  - Linear probe            : logistic regression on frozen train embeddings → test
  - LOPO                    : Leave-One-Patient-Out linear probe on full dataset
  - Retrieval               : Precision@K, Recall@K, mAP  (cosine similarity)
  - UMAP visualizations     : colored by label and by patient_id

All metrics are saved to <results_dir>/eval_metrics.json.
"""

import argparse
import base64
import json
import logging
import os

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    roc_auc_score,
    confusion_matrix,
)
import faiss
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import EmbeddingDataset
from model import DeepResidualEmbeddingModel
from visualization import visualize_umap, plot_evaluation_results, print_evaluation_metrics
from config import EVAL_CONFIG


# ============================================================================
# DATA LOADING
# ============================================================================

def _encode_patient_ids(series):
    """Map patient IDs (possibly strings like '01-001') to integer codes."""
    codes, _ = pd.factorize(series)
    return codes


def _per_patient_normalize(df, feature_cols, pid_col, label_col):
    """Z-score each feature relative to the patient's non-seizure baseline."""
    df = df.copy()
    for pid, grp in df.groupby(pid_col):
        baseline = grp[grp[label_col] == 0]
        if len(baseline) < 2:
            baseline = grp
        pmean = baseline[feature_cols].mean()
        pstd = baseline[feature_cols].std().replace(0, 1)
        df.loc[grp.index, feature_cols] = (grp[feature_cols] - pmean) / pstd
    return df


def load_data(config):
    """
    Reload train/test splits using the exact same preprocessing as training.
    Returns numpy arrays: train_feats, train_labels, train_pids, test_feats, test_labels, test_pids.
    """
    data_cfg = config["data_config"]
    pid_col = data_cfg.get("patient_id_col", "patient_id")
    label_col = data_cfg.get("label_col", "label")
    split_col = data_cfg.get("split_col", "split")
    avail_cols = data_cfg.get("availability_cols", [])

    df = pd.read_csv(data_cfg["data_path"])
    drop_cols = [c for c in data_cfg["drop_columns"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    train_df = df[df[split_col] == data_cfg["train_split_name"]].drop(columns=[split_col])
    test_df = df[df[split_col] == data_cfg["test_split_name"]].drop(columns=[split_col])

    if data_cfg["num_patients_train"] is not None:
        pids = train_df[pid_col].unique()
        train_df = train_df[train_df[pid_col].isin(pids[: data_cfg["num_patients_train"]])]

    non_feature = [label_col, pid_col] + [c for c in avail_cols if c in train_df.columns]
    feature_cols = [c for c in train_df.columns if c not in non_feature]

    # Drop constant features
    stds = train_df[feature_cols].std()
    constant = stds[stds == 0].index.tolist()
    if constant:
        feature_cols = [c for c in feature_cols if c not in constant]
        train_df.drop(columns=constant, inplace=True)
        test_df.drop(columns=constant, inplace=True)

    # Fill NaN
    train_df[feature_cols] = train_df[feature_cols].fillna(0)
    test_df[feature_cols] = test_df[feature_cols].fillna(0)

    # Normalisation
    if data_cfg.get("per_patient_normalization", False):
        train_df = _per_patient_normalize(train_df, feature_cols, pid_col, label_col)
        test_df = _per_patient_normalize(test_df, feature_cols, pid_col, label_col)
        train_df[feature_cols] = train_df[feature_cols].fillna(0)
        test_df[feature_cols] = test_df[feature_cols].fillna(0)
    else:
        mean = train_df[feature_cols].mean()
        std = train_df[feature_cols].std().replace(0, 1)
        train_df[feature_cols] = (train_df[feature_cols] - mean) / std
        test_df[feature_cols] = (test_df[feature_cols] - mean) / std

    def split_to_tensors(split_df):
        feats = torch.tensor(split_df[feature_cols].to_numpy(), dtype=torch.float32)
        labels = torch.tensor(split_df[label_col].to_numpy(), dtype=torch.long)
        pids = torch.tensor(_encode_patient_ids(split_df[pid_col]), dtype=torch.long)
        return feats, labels, pids

    return (*split_to_tensors(train_df), *split_to_tensors(test_df))


# ============================================================================
# EMBEDDING EXTRACTION
# ============================================================================

def extract_embeddings(model, features, labels, pids, device, batch_size=512):
    """Run the embedding model in eval mode and return (embeddings, labels, pids) as numpy arrays."""
    dataset = EmbeddingDataset(features, labels, pids)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    model.eval()
    all_embs, all_labels, all_pids = [], [], []

    with torch.no_grad():
        for batch_features, batch_labels, batch_pids in tqdm(loader, desc="Extracting embeddings", leave=False):
            embs = model(batch_features.to(device))
            all_embs.append(embs.cpu().numpy())
            all_labels.append(batch_labels.numpy())
            all_pids.append(batch_pids.numpy())

    return (
        np.concatenate(all_embs),
        np.concatenate(all_labels),
        np.concatenate(all_pids),
    )


# ============================================================================
# kNN CLASSIFIER
# ============================================================================

def _eval_knn_faiss(train_embs, train_labels, test_embs, test_labels, k=5, nprobe=32):
    """Fast approximate kNN using FAISS IVF index (cosine on L2-normed vectors)."""
    # L2-normalize so inner product == cosine similarity
    train_normed = train_embs / (np.linalg.norm(train_embs, axis=1, keepdims=True) + 1e-8)
    test_normed = test_embs / (np.linalg.norm(test_embs, axis=1, keepdims=True) + 1e-8)

    train_normed = np.ascontiguousarray(train_normed.astype(np.float32))
    test_normed = np.ascontiguousarray(test_normed.astype(np.float32))

    d = train_normed.shape[1]
    n_train = train_normed.shape[0]
    nlist = int(np.sqrt(n_train))

    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(train_normed)
    index.add(train_normed)
    index.nprobe = nprobe
    logging.info(f"FAISS IVF index: nlist={nlist}, nprobe={nprobe}")

    distances, indices = index.search(test_normed, k)

    # Majority vote
    neighbor_labels = train_labels[indices]  # (n_test, k)
    preds = np.array([np.bincount(row.astype(int), minlength=2).argmax() for row in neighbor_labels])

    # Probability estimate (fraction of positive neighbors)
    probs = neighbor_labels.astype(float).mean(axis=1)

    return preds, probs


def eval_knn(train_embs, train_labels, test_embs, test_labels, k=5):
    """kNN: use train embeddings as gallery, classify test queries by majority vote.
    Uses FAISS IVF index for fast approximate cosine search."""
    preds, probs = _eval_knn_faiss(train_embs, train_labels, test_embs, test_labels, k)

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, preds, average="binary", zero_division=0
    )
    try:
        roc_auc = float(roc_auc_score(test_labels, probs))
    except ValueError:
        roc_auc = None

    return {
        "accuracy": float(accuracy_score(test_labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
    }


# ============================================================================
# LINEAR PROBE
# ============================================================================

def eval_linear_probe(train_embs, train_labels, test_embs, test_labels):
    """Logistic regression trained on frozen train embeddings, evaluated on test."""
    clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    clf.fit(train_embs, train_labels)
    preds = clf.predict(test_embs)
    probs = clf.predict_proba(test_embs)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        test_labels, preds, average="binary", zero_division=0
    )
    try:
        roc_auc = float(roc_auc_score(test_labels, probs))
    except ValueError:
        roc_auc = None

    return {
        "accuracy": float(accuracy_score(test_labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
    }


# ============================================================================
# LOPO — LEAVE-ONE-PATIENT-OUT
# ============================================================================

def eval_lopo(all_embs, all_labels, all_pids):
    """
    For each patient: train a logistic regression on all other patients' embeddings,
    evaluate on the held-out patient. Uses the full dataset (train + test combined)
    so every patient is evaluated at least once.

    Returns macro metrics (concatenated predictions across all folds) and per-patient metrics.
    """
    unique_pids = np.unique(all_pids)
    per_patient = {}
    concat_preds, concat_true = [], []

    for pid in tqdm(unique_pids, desc="LOPO"):
        mask_test = all_pids == pid
        mask_train = ~mask_test

        # Skip degenerate folds
        if mask_train.sum() == 0 or mask_test.sum() == 0:
            continue
        if len(np.unique(all_labels[mask_train])) < 2:
            continue

        clf = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
        clf.fit(all_embs[mask_train], all_labels[mask_train])
        preds = clf.predict(all_embs[mask_test])

        concat_preds.extend(preds)
        concat_true.extend(all_labels[mask_test])

        # Per-patient metrics only when both classes are present in the test fold
        if len(np.unique(all_labels[mask_test])) >= 2:
            p, r, f1, _ = precision_recall_fscore_support(
                all_labels[mask_test], preds, average="binary", zero_division=0
            )
            per_patient[int(pid)] = {
                "accuracy": float(accuracy_score(all_labels[mask_test], preds)),
                "precision": float(p),
                "recall": float(r),
                "f1": float(f1),
                "n_samples": int(mask_test.sum()),
            }

    concat_preds = np.array(concat_preds)
    concat_true = np.array(concat_true)

    macro_p, macro_r, macro_f1, _ = precision_recall_fscore_support(
        concat_true, concat_preds, average="binary", zero_division=0
    )

    return {
        "macro_accuracy": float(accuracy_score(concat_true, concat_preds)),
        "macro_precision": float(macro_p),
        "macro_recall": float(macro_r),
        "macro_f1": float(macro_f1),
        "num_patients_evaluated": len(per_patient),
        "per_patient": per_patient,
    }


# ============================================================================
# RETRIEVAL METRICS
# ============================================================================

def eval_retrieval(query_embs, query_labels, gallery_embs, gallery_labels, k_values=(5, 10, 20)):
    """
    For each query (test sample), rank the gallery (train samples) by cosine similarity.
    A gallery item is "relevant" if it shares the same label as the query.

    Uses FAISS for top-K retrieval to avoid building the full similarity matrix.

    Metrics (computed per query, then averaged):
      - Precision@K: fraction of the top-K retrieved items that share the query's label.
        E.g. P@5=0.8 means 4 out of 5 nearest neighbors have the same label.
      - Recall@K: fraction of ALL same-label items in the gallery that appear in the top-K.
        E.g. if there are 1000 seizure items in the gallery and 3 appear in top-20, R@20=0.003.
      - mAP@K: average precision over the ranked top-K list — rewards relevant items
        appearing earlier in the ranking.
    """
    max_k = max(k_values)

    logging.info(f"Retrieval: normalizing {len(query_embs)} queries and {len(gallery_embs)} gallery vectors...")
    q_norm = query_embs / (np.linalg.norm(query_embs, axis=1, keepdims=True) + 1e-8)
    g_norm = gallery_embs / (np.linalg.norm(gallery_embs, axis=1, keepdims=True) + 1e-8)

    q_norm = np.ascontiguousarray(q_norm.astype(np.float32))
    g_norm = np.ascontiguousarray(g_norm.astype(np.float32))

    d = g_norm.shape[1]

    n_gallery = len(g_norm)
    nlist = int(np.sqrt(n_gallery))
    nprobe = 32
    logging.info(f"Retrieval: building FAISS IVF index (dim={d}, gallery={n_gallery}, nlist={nlist}, nprobe={nprobe})...")
    quantizer = faiss.IndexFlatIP(d)
    index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(g_norm)
    index.add(g_norm)
    index.nprobe = nprobe
    logging.info(f"Retrieval: searching top-{max_k} for {len(q_norm)} queries...")
    _, indices = index.search(q_norm, max_k)
    logging.info("Retrieval: FAISS search done, computing metrics...")

    # Pre-count relevant items per query label for recall
    label_counts = {}
    for lbl in np.unique(query_labels):
        label_counts[lbl] = int((gallery_labels == lbl).sum())

    prec_at_k = {k: [] for k in k_values}
    rec_at_k = {k: [] for k in k_values}
    ap_list = []

    for i in range(len(query_labels)):
        top_k_labels = gallery_labels[indices[i]]
        relevance = (top_k_labels == query_labels[i]).astype(float)
        n_relevant = label_counts.get(query_labels[i], 0)

        if n_relevant == 0:
            continue

        for k in k_values:
            top_k = relevance[:k]
            prec_at_k[k].append(top_k.mean())
            rec_at_k[k].append(top_k.sum() / n_relevant)

        # Average Precision (truncated at max_k)
        cumsum = np.cumsum(relevance)
        ranks = np.arange(1, max_k + 1)
        ap = ((cumsum / ranks) * relevance).sum() / n_relevant
        ap_list.append(ap)

    results = {}
    for k in k_values:
        results[f"precision@{k}"] = float(np.mean(prec_at_k[k])) if prec_at_k[k] else 0.0
        results[f"recall@{k}"] = float(np.mean(rec_at_k[k])) if rec_at_k[k] else 0.0
    results["mAP"] = float(np.mean(ap_list)) if ap_list else 0.0

    return results


# ============================================================================
# CLASSIFIER EVALUATION
# ============================================================================

def evaluate_classifier(classifier, dataloader, device):
    """Run the classifier on a dataloader and return metrics dict."""
    classifier.eval()
    all_preds, all_labels, all_probs = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            features, labels, _ = batch
            features, labels = features.to(device), labels.to(device)
            logits = classifier(features)
            probs = torch.softmax(logits, dim=1)
            all_preds.extend(torch.argmax(logits, dim=1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    try:
        roc_auc = float(roc_auc_score(all_labels, all_probs))
    except ValueError:
        roc_auc = None
        logging.warning("ROC-AUC could not be computed (only one class present)")

    return {
        "accuracy": float(accuracy_score(all_labels, all_preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(all_labels, all_preds),
        "predictions_probs": np.array(all_probs),
        "true_labels": np.array(all_labels),
    }


def eval_classifier_head(classifier, train_dataloader, test_dataloader, device, results_dir):
    """Evaluate the trained classifier on train and test sets and save plots."""
    train_metrics = None
    if EVAL_CONFIG["evaluate_on_train"]:
        logging.info("Evaluating classifier on training set...")
        train_metrics = evaluate_classifier(classifier, train_dataloader, device)
        print_evaluation_metrics(train_metrics, dataset_name="Train")
        plot_evaluation_results(
            train_metrics, train_metrics["predictions_probs"], train_metrics["true_labels"],
            save_path=os.path.join(results_dir, "train_evaluation_results.png"),
        )

    test_metrics = None
    if EVAL_CONFIG["evaluate_on_test"]:
        logging.info("Evaluating classifier on test set...")
        test_metrics = evaluate_classifier(classifier, test_dataloader, device)
        print_evaluation_metrics(test_metrics, dataset_name="Test")
        plot_evaluation_results(
            test_metrics, test_metrics["predictions_probs"], test_metrics["true_labels"],
            save_path=os.path.join(results_dir, "test_evaluation_results.png"),
        )

    return train_metrics, test_metrics


# ============================================================================
# UMAP VISUALIZATION
# ============================================================================

def _extract_from_dataloader(embedding_model, dataloader, device, sample_size=None):
    """Extract embeddings, labels, patient_ids from a dataloader."""
    embedding_model.eval()
    encoded_data, labels, patient_ids = [], [], []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings", leave=False):
            if len(batch) == 3:       # EmbeddingDataset
                features, label, patient_id = batch
            elif len(batch) == 4:     # ContrastiveDataset
                features, _, _, label = batch
                patient_id = torch.zeros_like(label)
            elif len(batch) == 5:     # TripletDataset
                features, _, _, label, patient_id = batch
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")

            embeddings = embedding_model(features.to(device))
            encoded_data.extend(embeddings.cpu().numpy())
            labels.extend(label.cpu().numpy())
            patient_ids.extend(patient_id.cpu().numpy())

    encoded_data = np.array(encoded_data)
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)

    if sample_size is not None and len(encoded_data) > sample_size:
        # Keep all seizure samples (up to sample_size), fill the rest with no-seizure.
        # Cap seizure at 10% of sample_size to avoid distorting UMAP neighborhoods.
        minority_idx = np.where(labels == 1)[0]
        majority_idx = np.where(labels != 1)[0]

        max_minority = min(len(minority_idx), sample_size // 10)
        if len(minority_idx) > max_minority:
            minority_idx = np.random.choice(minority_idx, max_minority, replace=False)

        n_majority = sample_size - len(minority_idx)
        if n_majority < len(majority_idx):
            majority_idx = np.random.choice(majority_idx, n_majority, replace=False)

        idx = np.concatenate([minority_idx, majority_idx])
        np.random.shuffle(idx)
        encoded_data, labels, patient_ids = encoded_data[idx], labels[idx], patient_ids[idx]

    return encoded_data, labels, patient_ids


def generate_umap_visualizations(embedding_model, dataloader, device, results_dir,
                                 sample_size=None, prefix='umap', test_dataloader=None):
    """Generate UMAP plots. UMAP is fit on train data; test is projected into the same space."""
    logging.info(f"Generating UMAP visualizations ({prefix})...")

    encoded_data, labels, patient_ids = _extract_from_dataloader(
        embedding_model, dataloader, device, sample_size)

    test_data, test_labels, test_pids = None, None, None
    if test_dataloader is not None:
        test_data, test_labels, test_pids = _extract_from_dataloader(
            embedding_model, test_dataloader, device, sample_size)

    visualize_umap(encoded_data, labels, patient_ids, results_dir, prefix=prefix,
                   test_data=test_data, test_labels=test_labels, test_patient_ids=test_pids)
    logging.info("UMAP visualizations completed!\n")


# ============================================================================
# HTML REPORT
# ============================================================================

def _img_tag(path, title=""):
    """Return an <img> tag with the image embedded as base64, or empty string if file missing."""
    if not os.path.exists(path):
        return ""
    with open(path, "rb") as f:
        data = base64.b64encode(f.read()).decode()
    return (
        f'<figure>'
        f'<img src="data:image/png;base64,{data}" alt="{title}">'
        f'<figcaption>{title}</figcaption>'
        f'</figure>'
    )


def _metrics_table(metrics, exclude=("per_patient",)):
    """Render a flat dict of metrics as an HTML table row, skipping excluded keys."""
    rows = ""
    for k, v in metrics.items():
        if k in exclude:
            continue
        if isinstance(v, float):
            rows += f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
        else:
            rows += f"<tr><td>{k}</td><td>{v}</td></tr>"
    return f'<table><tr><th>Metric</th><th>Value</th></tr>{rows}</table>'


def _config_table(cfg):
    """Render a config dict as an HTML table, skipping nested dicts."""
    rows = ""
    for k, v in cfg.items():
        if not isinstance(v, dict):
            rows += f"<tr><td>{k}</td><td>{v}</td></tr>"
    return f'<table><tr><th>Parameter</th><th>Value</th></tr>{rows}</table>'


def generate_report(results_dir):
    """
    Build a self-contained HTML report from eval_metrics.json, training_config.json,
    and all PNG images found in results_dir. Saved as report.html.
    """
    # Load data
    metrics_path = os.path.join(results_dir, "eval_metrics.json")
    config_path = os.path.join(results_dir, "training_config.json")

    metrics = json.load(open(metrics_path)) if os.path.exists(metrics_path) else {}
    config = json.load(open(config_path)) if os.path.exists(config_path) else {}

    timestamp = config.get("timestamp", "")
    model_cfg = config.get("model_config", {})
    emb_cfg = config.get("embedding_training_config", {})
    clf_cfg = config.get("classifier_training_config", {})

    def section(title, content):
        return f'<section><h2>{title}</h2>{content}</section>'

    def img_row(*paths_titles):
        imgs = "".join(_img_tag(os.path.join(results_dir, p), t) for p, t in paths_titles if p)
        return f'<div class="img-row">{imgs}</div>' if imgs else ""

    # --- Config section ---
    config_html = (
        "<h3>Model</h3>" + _config_table(model_cfg) +
        "<h3>Embedding Training</h3>" + _config_table(emb_cfg) +
        "<h3>Classifier Training</h3>" + _config_table(clf_cfg)
    )

    # --- Embedding metrics section ---
    emb_metrics_html = ""
    for key, title in [
        ("knn", "kNN Classifier"),
        ("linear_probe", "Linear Probe"),
    ]:
        if key in metrics:
            emb_metrics_html += f"<h3>{title}</h3>" + _metrics_table(metrics[key])

    if "lopo" in metrics:
        lopo = metrics["lopo"]
        macro = {k: v for k, v in lopo.items() if k != "per_patient"}
        emb_metrics_html += "<h3>LOPO (Leave-One-Patient-Out)</h3>" + _metrics_table(macro)

        if lopo.get("per_patient"):
            rows = "".join(
                f"<tr><td>{pid}</td><td>{v['accuracy']:.4f}</td><td>{v['precision']:.4f}</td>"
                f"<td>{v['recall']:.4f}</td><td>{v['f1']:.4f}</td><td>{v['n_samples']}</td></tr>"
                for pid, v in lopo["per_patient"].items()
            )
            emb_metrics_html += (
                "<h4>Per-patient breakdown</h4>"
                "<table><tr><th>Patient</th><th>Accuracy</th><th>Precision</th>"
                "<th>Recall</th><th>F1</th><th>Samples</th></tr>"
                f"{rows}</table>"
            )

    if "retrieval" in metrics:
        emb_metrics_html += "<h3>Retrieval</h3>" + _metrics_table(metrics["retrieval"])

    # --- Images ---
    umap_html = ""
    # Determine which model eval.py used
    finetuned_exists = os.path.exists(os.path.join(results_dir, "embedding_model_finetuned.pth"))
    eval_model_label = "fine-tuned embedding" if finetuned_exists else "Phase 1 embedding (no fine-tuning)"

    for phase_prefix, phase_title in [
        ("umap_pretrain", "Pre-fine-tune (Phase 1 embedding)"),
        ("umap_finetuned", "Post-fine-tune (Phase 2 — unfrozen embedding)"),
        ("umap_eval", f"Eval ({eval_model_label})"),
    ]:
        train_exists = any(os.path.exists(os.path.join(results_dir, f)) for f in (
            f"{phase_prefix}_labels.png", f"{phase_prefix}_patient_ids.png"
        ))
        test_exists = any(os.path.exists(os.path.join(results_dir, f)) for f in (
            f"{phase_prefix}_test_labels.png", f"{phase_prefix}_test_patient_ids.png"
        ))
        if train_exists:
            umap_html += f"<h3>{phase_title} — Train</h3>"
            umap_html += img_row(
                (f"{phase_prefix}_labels.png", "by label"),
                (f"{phase_prefix}_patient_ids.png", "by patient"),
            )
        if test_exists:
            umap_html += f"<h3>{phase_title} — Test (same UMAP space)</h3>"
            umap_html += img_row(
                (f"{phase_prefix}_test_labels.png", "by label"),
                (f"{phase_prefix}_test_patient_ids.png", "by patient"),
            )
    classifier_html = img_row(
        ("classifier_training_history.png", "Classifier training history"),
    ) + img_row(
        ("train_evaluation_results.png", "Train — ROC & confusion matrix"),
        ("test_evaluation_results.png", "Test — ROC & confusion matrix"),
    )

    # --- Assemble page ---
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Evaluation Report — {os.path.basename(results_dir)}</title>
<style>
  body {{ font-family: sans-serif; max-width: 1200px; margin: 40px auto; padding: 0 20px; color: #222; }}
  h1 {{ border-bottom: 2px solid #444; padding-bottom: 8px; }}
  h2 {{ margin-top: 40px; border-bottom: 1px solid #ccc; padding-bottom: 4px; color: #333; }}
  h3 {{ margin-top: 24px; color: #555; }}
  h4 {{ margin-top: 16px; color: #666; }}
  table {{ border-collapse: collapse; margin: 12px 0; min-width: 320px; }}
  th, td {{ border: 1px solid #ddd; padding: 7px 14px; text-align: left; }}
  th {{ background: #f4f4f4; font-weight: 600; }}
  tr:nth-child(even) {{ background: #fafafa; }}
  .img-row {{ display: flex; flex-wrap: wrap; gap: 24px; margin: 16px 0; }}
  figure {{ margin: 0; }}
  figcaption {{ text-align: center; font-size: 0.85em; color: #666; margin-top: 6px; }}
  img {{ max-width: 560px; width: 100%; border: 1px solid #ddd; border-radius: 4px; }}
  .meta {{ color: #666; font-size: 0.9em; margin-bottom: 24px; }}
</style>
</head>
<body>
<h1>Evaluation Report</h1>
<p class="meta">Results directory: <code>{results_dir}</code> &nbsp;|&nbsp; Trained: {timestamp}</p>

{section("Training Configuration", config_html)}
{section("Embedding Evaluation Metrics", emb_metrics_html)}
{section("UMAP Visualizations", umap_html)}
{section("Classifier Evaluation", classifier_html)}

</body>
</html>"""

    out_path = os.path.join(results_dir, "report.html")
    with open(out_path, "w") as f:
        f.write(html)
    logging.info(f"Report saved to {out_path}")
    return out_path


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate a trained contrastive embedding model")
    parser.add_argument(
        "--results_dir", required=True,
        help="Path to results folder containing training_config.json (and optionally the .pth model)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Path to the .pth model file. Auto-detected inside --results_dir if omitted."
    )
    parser.add_argument("--knn_k", type=int, default=5, help="k for kNN classifier (default: 5)")
    parser.add_argument(
        "--retrieval_k", type=int, nargs="+", default=[5, 10, 20],
        help="K values for retrieval metrics (default: 5 10 20)"
    )
    parser.add_argument(
        "--umap_sample_size", type=int, default=2000,
        help="Max samples for UMAP visualizations (default: 2000)"
    )
    parser.add_argument("--no_umap", action="store_true", help="Skip UMAP generation")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

    # --- Load training config ---
    config_path = os.path.join(args.results_dir, "training_config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"training_config.json not found in {args.results_dir}")
    with open(config_path) as f:
        config = json.load(f)

    # --- Locate model weights ---
    # Prefer the fine-tuned embedding (Phase 2) over the Phase 1 checkpoint.
    model_path = args.model
    if model_path is None:
        for candidate in ("embedding_model_finetuned.pth", "embedding_model.pth"):
            candidate_path = os.path.join(args.results_dir, candidate)
            if os.path.exists(candidate_path):
                model_path = candidate_path
                break
        if model_path is None:
            raise FileNotFoundError(
                f"No .pth file found in {args.results_dir}. Pass --model explicitly."
            )
        logging.info(f"Auto-detected model: {model_path}")

    # --- Device ---
    device_cfg = config.get("device_config", {})
    if device_cfg.get("use_cuda", True) and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_cfg.get('cuda_device', 0)}")
    else:
        device = torch.device("cpu")
    logging.info(f"Using device: {device}")

    # --- Load data ---
    logging.info("Loading data...")
    train_feats, train_labels, train_pids, test_feats, test_labels, test_pids = load_data(config)

    # --- Build and load model ---
    model_cfg = config["model_config"]
    # Auto-detect input_dim from loaded data if not saved in config
    input_dim = model_cfg.get("input_dim") or train_feats.shape[1]
    model = DeepResidualEmbeddingModel(
        input_dim=input_dim,
        embedding_dim=model_cfg["embedding_dim"],
        num_blocks=model_cfg.get("num_residual_blocks", 3),
        dropout=model_cfg.get("embedding_dropout", 0.2),
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    logging.info(f"Loaded model from {model_path} (input_dim={input_dim})")

    # --- Extract embeddings ---
    logging.info("Extracting embeddings...")
    train_embs, train_labels_np, train_pids_np = extract_embeddings(
        model, train_feats, train_labels, train_pids, device
    )
    test_embs, test_labels_np, test_pids_np = extract_embeddings(
        model, test_feats, test_labels, test_pids, device
    )
    logging.info(f"Train embeddings: {train_embs.shape}, Test embeddings: {test_embs.shape}")

    all_results = {}

    # --- kNN ---
    logging.info(f"\n{'='*60}\nkNN Classifier  (k={args.knn_k})\n{'='*60}")
    knn_metrics = eval_knn(train_embs, train_labels_np, test_embs, test_labels_np, k=args.knn_k)
    all_results["knn"] = knn_metrics
    for metric, val in knn_metrics.items():
        logging.info(f"  {metric:<12}: {val:.4f}" if val is not None else f"  {metric:<12}: N/A")

    # --- Linear probe ---
    logging.info(f"\n{'='*60}\nLinear Probe\n{'='*60}")
    linear_metrics = eval_linear_probe(train_embs, train_labels_np, test_embs, test_labels_np)
    all_results["linear_probe"] = linear_metrics
    for metric, val in linear_metrics.items():
        logging.info(f"  {metric:<12}: {val:.4f}" if val is not None else f"  {metric:<12}: N/A")

    # --- LOPO ---
    logging.info(f"\n{'='*60}\nLOPO (Leave-One-Patient-Out)\n{'='*60}")
    all_embs = np.concatenate([train_embs, test_embs])
    all_labels_full = np.concatenate([train_labels_np, test_labels_np])
    all_pids_full = np.concatenate([train_pids_np, test_pids_np])
    lopo_metrics = eval_lopo(all_embs, all_labels_full, all_pids_full)
    all_results["lopo"] = lopo_metrics
    logging.info(f"  Patients evaluated : {lopo_metrics['num_patients_evaluated']}")
    logging.info(f"  Macro accuracy     : {lopo_metrics['macro_accuracy']:.4f}")
    logging.info(f"  Macro precision    : {lopo_metrics['macro_precision']:.4f}")
    logging.info(f"  Macro recall       : {lopo_metrics['macro_recall']:.4f}")
    logging.info(f"  Macro F1           : {lopo_metrics['macro_f1']:.4f}")

    # --- Retrieval ---
    logging.info(f"\n{'='*60}\nRetrieval  (K={args.retrieval_k})\n{'='*60}")
    retrieval_metrics = eval_retrieval(
        test_embs, test_labels_np, train_embs, train_labels_np, k_values=args.retrieval_k
    )
    all_results["retrieval"] = retrieval_metrics
    for metric, val in retrieval_metrics.items():
        logging.info(f"  {metric:<15}: {val:.4f}")

    # --- Save metrics ---
    out_path = os.path.join(args.results_dir, "eval_metrics.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=4)
    logging.info(f"\nMetrics saved to {out_path}")

    # --- UMAP (only if model differs from Phase 1, i.e. fine-tuned embedding exists) ---
    finetuned_exists = os.path.exists(os.path.join(args.results_dir, "embedding_model_finetuned.pth"))
    if not args.no_umap and finetuned_exists:
        logging.info(f"\n{'='*60}\nUMAP Visualizations (fine-tuned embedding)\n{'='*60}")
        sample = args.umap_sample_size

        def _subsample_keep_minority(embs, labels, pids, n):
            if n is None or len(embs) <= n:
                return embs, labels, pids
            minority_idx = np.where(labels == 1)[0]
            majority_idx = np.where(labels != 1)[0]
            max_minority = min(len(minority_idx), n // 10)
            if len(minority_idx) > max_minority:
                minority_idx = np.random.choice(minority_idx, max_minority, replace=False)
            n_majority = n - len(minority_idx)
            if n_majority < len(majority_idx):
                majority_idx = np.random.choice(majority_idx, n_majority, replace=False)
            idx = np.concatenate([minority_idx, majority_idx])
            np.random.shuffle(idx)
            return embs[idx], labels[idx], pids[idx]

        train_viz, train_lbl_viz, train_pid_viz = _subsample_keep_minority(
            train_embs, train_labels_np, train_pids_np, sample)
        test_viz, test_lbl_viz, test_pid_viz = _subsample_keep_minority(
            test_embs, test_labels_np, test_pids_np, sample)

        visualize_umap(train_viz, train_lbl_viz, train_pid_viz, args.results_dir,
                       prefix='umap_eval',
                       test_data=test_viz, test_labels=test_lbl_viz, test_patient_ids=test_pid_viz)
    elif not args.no_umap:
        logging.info("Skipping eval UMAP — no fine-tuned embedding, same as Phase 1 pretrain UMAP")

    # --- HTML report ---
    generate_report(args.results_dir)

    logging.info("\nEvaluation complete!")


if __name__ == "__main__":
    main()
