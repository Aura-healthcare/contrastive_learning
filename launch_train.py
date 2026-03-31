"""
launch_train.py — Train a contrastive embedding model then fine-tune a classifier.

Embedding loss is controlled by EMBEDDING_TRAINING_CONFIG['loss_type'] in config.py:
  'contrastive'        — pairwise contrastive loss (ContrastiveDataset, cross-patient pairs)
  'triplet'            — offline triplet loss (TripletDataset, cross-patient positives)
  'batch_hard_triplet' — online hard mining (EmbeddingDataset + BatchHardTripletLoss)
"""

import pandas as pd
import logging
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import json
from datetime import datetime

from model import DeepResidualEmbeddingModel, SeizureClassifier
from loss import ContrastiveLoss, TripletLoss, BatchHardTripletLoss, SupervisedContrastiveLoss
from dataset import ContrastiveDataset, TripletDataset, EmbeddingDataset, PKBatchSampler
from train import (
    train_model,
    train_model_triplet,
    train_model_triplet_hard_negative,
    train_model_supcon,
    train_classifier,
)
from visualization import plot_training_history
from eval import generate_umap_visualizations, eval_classifier_head
from config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    EMBEDDING_TRAINING_CONFIG,
    CLASSIFIER_TRAINING_CONFIG,
    EVAL_CONFIG,
    DEVICE_CONFIG,
    LOGGING_CONFIG,
)


# ============================================================================
# CONFIGURATION SAVING
# ============================================================================

def save_config_to_json(results_dir):
    config_dict = {
        'data_config': DATA_CONFIG,
        'model_config': MODEL_CONFIG,
        'embedding_training_config': EMBEDDING_TRAINING_CONFIG,
        'classifier_training_config': CLASSIFIER_TRAINING_CONFIG,
        'eval_config': EVAL_CONFIG,
        'device_config': DEVICE_CONFIG,
        'logging_config': LOGGING_CONFIG,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path = os.path.join(results_dir, 'training_config.json')
    with open(config_path, 'w') as f:
        json.dump(config_dict, f, indent=4)
    logging.info(f"Configuration saved to {config_path}")
    return config_path


# ============================================================================
# DATA LOADING
# ============================================================================

def load_train_test_dataset(csv_path, loss_type='batch_hard_triplet'):
    """Load and preprocess the cardiac features dataset."""
    logging.info(f"Loading dataset from {csv_path}")
    logging.info(f"Using loss type: {loss_type}")
    df = pd.read_csv(csv_path)

    df.drop(columns=DATA_CONFIG['drop_columns'], inplace=True)

    train_df = df[df['split'] == DATA_CONFIG['train_split_name']].drop(columns=['split'])
    test_df = df[df['split'] == DATA_CONFIG['test_split_name']].drop(columns=['split'])

    if DATA_CONFIG['num_patients_train'] is not None:
        patient_ids_list = train_df['patient_id'].unique()
        train_df = train_df[train_df['patient_id'].isin(
            patient_ids_list[:DATA_CONFIG['num_patients_train']]
        )]

    logging.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    logging.info(f"Train label distribution:\n{train_df['label'].value_counts()}")
    logging.info(f"Test label distribution:\n{test_df['label'].value_counts()}")

    feature_columns = train_df.drop(columns=['label', 'patient_id']).columns
    train_mean = train_df[feature_columns].mean()
    train_std = train_df[feature_columns].std().replace(0, 1)

    train_df[feature_columns] = (train_df[feature_columns] - train_mean) / train_std
    test_df[feature_columns] = (test_df[feature_columns] - train_mean) / train_std

    train_features = torch.tensor(train_df.drop(columns=['label', 'patient_id']).to_numpy(), dtype=torch.float32)
    train_labels = torch.tensor(train_df['label'].to_numpy(), dtype=torch.long)
    train_patient_ids = torch.tensor(train_df['patient_id'].to_numpy(), dtype=torch.long)

    test_features = torch.tensor(test_df.drop(columns=['label', 'patient_id']).to_numpy(), dtype=torch.float32)
    test_labels = torch.tensor(test_df['label'].to_numpy(), dtype=torch.long)
    test_patient_ids = torch.tensor(test_df['patient_id'].to_numpy(), dtype=torch.long)

    if loss_type == 'contrastive':
        train_dataset = ContrastiveDataset(
            train_features,
            train_labels,
            patient_ids=train_patient_ids,
            num_pairs=EMBEDDING_TRAINING_CONFIG['num_pairs'],
        )
        test_dataset = ContrastiveDataset(
            test_features,
            test_labels,
            patient_ids=test_patient_ids,
            num_pairs=1000,
        )
    elif loss_type == 'triplet':
        train_dataset = TripletDataset(train_features, train_labels, train_patient_ids)
        test_dataset = TripletDataset(test_features, test_labels, test_patient_ids)
    else:  # 'batch_hard_triplet', 'supcon', or 'simple'
        train_dataset = EmbeddingDataset(train_features, train_labels, train_patient_ids)
        test_dataset = EmbeddingDataset(test_features, test_labels, test_patient_ids)

    patient_counts = train_df['patient_id'].value_counts()
    weights = train_df['patient_id'].apply(lambda x: 1.0 / patient_counts[x]).values

    return train_dataset, test_dataset, weights


# ============================================================================
# EMBEDDING TRAINING
# ============================================================================

def train_embedding_model(train_dataloader, device, results_dir=None):
    """Train the embedding model using the loss type set in config."""
    loss_type = EMBEDDING_TRAINING_CONFIG['loss_type']

    logging.info("=" * 80)
    logging.info(f"PHASE 1: Training Embedding Model with {loss_type.upper()} Loss")
    logging.info("=" * 80)

    embedding_model = DeepResidualEmbeddingModel(
        input_dim=MODEL_CONFIG['input_dim'],
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        num_blocks=MODEL_CONFIG['num_residual_blocks'],
    ).to(device)

    if loss_type == 'contrastive':
        loss_fn = ContrastiveLoss()
    elif loss_type == 'triplet':
        loss_fn = TripletLoss(margin=EMBEDDING_TRAINING_CONFIG['margin'])
    elif loss_type == 'supcon':
        loss_fn = SupervisedContrastiveLoss(temperature=EMBEDDING_TRAINING_CONFIG['temperature'])
    else:  # batch_hard_triplet
        loss_fn = BatchHardTripletLoss(margin=EMBEDDING_TRAINING_CONFIG['margin'])

    optimizer = optim.Adam(
        embedding_model.parameters(),
        lr=EMBEDDING_TRAINING_CONFIG['learning_rate'],
    )

    scheduler = None
    if EMBEDDING_TRAINING_CONFIG['use_scheduler']:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=EMBEDDING_TRAINING_CONFIG['scheduler_step_size'],
            gamma=EMBEDDING_TRAINING_CONFIG['scheduler_gamma'],
        )

    os.makedirs(DATA_CONFIG['checkpoint_dir'], exist_ok=True)

    train_fn = {
        'contrastive': train_model,
        'triplet': train_model_triplet,
        'batch_hard_triplet': train_model_triplet_hard_negative,
        'supcon': train_model_supcon,
    }[loss_type]

    embedding_model, _ = train_fn(
        embedding_model,
        train_dataloader,
        optimizer,
        loss_fn,
        device,
        epochs=EMBEDDING_TRAINING_CONFIG['epochs'],
        scheduler=scheduler,
        checkpoint_dir=DATA_CONFIG['checkpoint_dir'],
    )

    logging.info("Embedding model training completed!\n")

    if results_dir is not None:
        model_save_path = os.path.join(results_dir, 'embedding_model.pth')
        torch.save(embedding_model.state_dict(), model_save_path)
        logging.info(f"Embedding model saved to {model_save_path}")

    return embedding_model


# ============================================================================
# CLASSIFICATION
# ============================================================================

def train_classifier_head(embedding_model, train_dataloader, test_dataloader, device, results_dir):
    """Train the classifier head on learned embeddings."""
    logging.info("=" * 80)
    logging.info("PHASE 2: Training Seizure Classifier")
    logging.info("=" * 80)

    classifier = SeizureClassifier(
        embedding_model=embedding_model,
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        num_classes=MODEL_CONFIG['num_classes'],
        hidden_dims=MODEL_CONFIG['classifier_hidden_dims'],
    ).to(device)

    if not CLASSIFIER_TRAINING_CONFIG['freeze_embeddings']:
        n_blocks = CLASSIFIER_TRAINING_CONFIG.get('unfreeze_last_n_blocks', None)
        if n_blocks is None:
            logging.info("Unfreezing ALL embedding layers for fine-tuning...")
        else:
            logging.info(f"Unfreezing last {n_blocks} residual blocks for fine-tuning...")
        classifier.unfreeze_embedding_model(last_n_blocks=n_blocks)

    # Class-weighted loss to handle seizure imbalance
    all_labels = []
    for batch in train_dataloader:
        if len(batch) == 3:
            _, labels, _ = batch
        elif len(batch) == 4:
            _, _, _, labels = batch
        elif len(batch) == 5:
            _, _, _, labels, _ = batch
        all_labels.extend(labels.numpy())

    class_counts = np.bincount(all_labels)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
    logging.info(f"Class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        classifier.parameters() if not CLASSIFIER_TRAINING_CONFIG['freeze_embeddings']
        else classifier.classifier.parameters(),
        lr=CLASSIFIER_TRAINING_CONFIG['learning_rate'],
    )

    scheduler = None
    if CLASSIFIER_TRAINING_CONFIG['use_scheduler']:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=CLASSIFIER_TRAINING_CONFIG['scheduler_step_size'],
            gamma=CLASSIFIER_TRAINING_CONFIG['scheduler_gamma'],
        )

    classifier, train_history = train_classifier(
        classifier, train_dataloader, optimizer, criterion, device,
        epochs=CLASSIFIER_TRAINING_CONFIG['epochs'], scheduler=scheduler,
    )
    logging.info("Classifier training completed!\n")

    plot_training_history(
        train_history,
        save_path=os.path.join(results_dir, 'classifier_training_history.png'),
    )

    # Save the (possibly fine-tuned) embedding model so eval.py picks up the
    # latest weights rather than the pre-fine-tune Phase 1 checkpoint.
    if not CLASSIFIER_TRAINING_CONFIG['freeze_embeddings']:
        finetuned_path = os.path.join(results_dir, 'embedding_model_finetuned.pth')
        torch.save(classifier.embedding_model.state_dict(), finetuned_path)
        logging.info(f"Fine-tuned embedding saved to {finetuned_path}")

    return classifier


# ============================================================================
# MAIN
# ============================================================================

def main():
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format'],
    )

    device = (
        torch.device(f"cuda:{DEVICE_CONFIG['cuda_device']}")
        if DEVICE_CONFIG['use_cuda'] and torch.cuda.is_available()
        else torch.device('cpu')
    )
    logging.info(f"Using device: {device}\n")

    if DATA_CONFIG['results_dir'] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"./results_{timestamp}"
    else:
        results_dir = DATA_CONFIG['results_dir']

    os.makedirs(results_dir, exist_ok=True)
    logging.info(f"Results will be saved to: {results_dir}\n")

    save_config_to_json(results_dir)

    loss_type = EMBEDDING_TRAINING_CONFIG['loss_type']

    # Dataset for embedding training
    train_dataset_embedding, _, weights = load_train_test_dataset(
        csv_path=DATA_CONFIG['data_path'],
        loss_type=loss_type,
    )

    # Dataset for classifier training and UMAP (always plain features/labels/pids)
    train_dataset_classifier, test_dataset_classifier, _ = load_train_test_dataset(
        csv_path=DATA_CONFIG['data_path'],
        loss_type='simple',
    )

    # Dataloader for embedding training
    if loss_type == 'supcon':
        # PKBatchSampler guarantees P patients × K samples per batch so that
        # every anchor has same-label positives from different patients — critical for SupCon.
        pk_sampler = PKBatchSampler(
            train_dataset_embedding.patient_ids,
            P=EMBEDDING_TRAINING_CONFIG['pk_p'],
            K=EMBEDDING_TRAINING_CONFIG['pk_k'],
            labels=train_dataset_embedding.labels,  # class-balanced batches for SupCon
        )
        train_dataloader_embedding = DataLoader(
            train_dataset_embedding,
            batch_sampler=pk_sampler,
        )
    elif DATA_CONFIG['use_weighted_sampling'] and loss_type != 'contrastive':
        # ContrastiveDataset generates pairs internally; WeightedRandomSampler doesn't apply.
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_dataloader_embedding = DataLoader(
            train_dataset_embedding,
            batch_size=EMBEDDING_TRAINING_CONFIG['batch_size'],
            sampler=sampler,
        )
    else:
        train_dataloader_embedding = DataLoader(
            train_dataset_embedding,
            batch_size=EMBEDDING_TRAINING_CONFIG['batch_size'],
            shuffle=True,
        )

    train_dataloader_classifier = DataLoader(
        train_dataset_classifier,
        batch_size=CLASSIFIER_TRAINING_CONFIG['batch_size'],
        shuffle=True,
    )
    test_dataloader_classifier = DataLoader(
        test_dataset_classifier,
        batch_size=CLASSIFIER_TRAINING_CONFIG['batch_size'],
        shuffle=False,
    )
    train_dataloader_umap = DataLoader(
        train_dataset_classifier,
        batch_size=128,
        shuffle=False,
    )

    # Phase 1: Train embedding model
    embedding_model = train_embedding_model(
        train_dataloader_embedding, device, results_dir=results_dir
    )

    # UMAP
    if EVAL_CONFIG['generate_umap']:
        generate_umap_visualizations(
            embedding_model, train_dataloader_umap, device, results_dir,
            sample_size=EVAL_CONFIG['umap_sample_size'],
            prefix='umap_pretrain',
        )

    # Phase 2: Train classifier
    classifier = train_classifier_head(
        embedding_model,
        train_dataloader_classifier,
        test_dataloader_classifier,
        device,
        results_dir,
    )

    # Evaluate classifier
    _, test_metrics = eval_classifier_head(
        classifier,
        train_dataloader_classifier,
        test_dataloader_classifier,
        device,
        results_dir,
    )

    logging.info("=" * 80)
    logging.info("TRAINING COMPLETE!")
    logging.info("=" * 80)
    if test_metrics is not None:
        logging.info(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
        logging.info(f"Final Test F1 Score: {test_metrics['f1']:.4f}")
        if test_metrics['roc_auc'] is not None:
            logging.info(f"Final Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logging.info(f"\nAll results saved to: {results_dir}")
    logging.info("Generated files: training_config.json, embedding_model.pth, "
                 "classifier_training_history.png, *_evaluation_results.png, umap_plot_*.png")


if __name__ == "__main__":
    main()
