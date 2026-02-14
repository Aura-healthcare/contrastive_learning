"""
Contrastive Learning for Seizure Detection from Cardiac Features

This script trains an embedding model using contrastive learning (batch hard triplet loss)
and then trains a classifier on top of the learned embeddings to predict seizure/no-seizure.
"""

import pandas as pd
import logging
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.optim as optim
import torch.nn as nn
from functools import lru_cache
import numpy as np
import os
import json
from datetime import datetime
from tqdm import tqdm

from model import DeepResidualEmbeddingModel, SeizureClassifier
from loss import ContrastiveLoss, TripletLoss, BatchHardTripletLoss
from dataset import ContrastiveDataset, TripletDataset, EmbeddingDataset
from train import (
    train_model,
    train_model_triplet,
    train_model_triplet_hard_negative,
    train_classifier,
    evaluate_classifier
)
from visualization import (
    plot_evaluation_results,
    plot_training_history,
    visualize_umap,
    print_evaluation_metrics
)
from config import (
    DATA_CONFIG,
    MODEL_CONFIG,
    EMBEDDING_TRAINING_CONFIG,
    CLASSIFIER_TRAINING_CONFIG,
    EVAL_CONFIG,
    DEVICE_CONFIG,
    LOGGING_CONFIG
)


# ============================================================================
# CONFIGURATION SAVING
# ============================================================================

def save_config_to_json(results_dir):
    """Save all configuration to a JSON file in the results directory."""
    config_dict = {
        'data_config': DATA_CONFIG,
        'model_config': MODEL_CONFIG,
        'embedding_training_config': EMBEDDING_TRAINING_CONFIG,
        'classifier_training_config': CLASSIFIER_TRAINING_CONFIG,
        'eval_config': EVAL_CONFIG,
        'device_config': DEVICE_CONFIG,
        'logging_config': LOGGING_CONFIG,
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

    # Drop unnecessary columns
    df.drop(columns=DATA_CONFIG['drop_columns'], inplace=True)

    # Split into train and test
    train_df = df[df['split'] == DATA_CONFIG['train_split_name']].drop(columns=['split'])
    test_df = df[df['split'] == DATA_CONFIG['test_split_name']].drop(columns=['split'])

    # Limit to subset of patients for faster experimentation
    if DATA_CONFIG['num_patients_train'] is not None:
        patient_ids_list = train_df['patient_id'].unique()
        train_df = train_df[train_df['patient_id'].isin(
            patient_ids_list[:DATA_CONFIG['num_patients_train']]
        )]

    logging.info(f"Train samples: {len(train_df)}, Test samples: {len(test_df)}")
    logging.info(f"Train label distribution:\n{train_df['label'].value_counts()}")
    logging.info(f"Test label distribution:\n{test_df['label'].value_counts()}")

    # Normalize features using train statistics
    feature_columns = train_df.drop(columns=['label', 'patient_id']).columns
    train_mean = train_df[feature_columns].mean()
    train_std = train_df[feature_columns].std()

    train_df[feature_columns] = (train_df[feature_columns] - train_mean) / train_std
    test_df[feature_columns] = (test_df[feature_columns] - train_mean) / train_std

    # Convert to tensors
    train_features = torch.tensor(train_df.drop(columns=['label', 'patient_id']).to_numpy(), dtype=torch.float32)
    train_labels = torch.tensor(train_df['label'].to_numpy(), dtype=torch.long)
    train_patient_ids = torch.tensor(train_df['patient_id'].to_numpy(), dtype=torch.long)

    test_features = torch.tensor(test_df.drop(columns=['label', 'patient_id']).to_numpy(), dtype=torch.float32)
    test_labels = torch.tensor(test_df['label'].to_numpy(), dtype=torch.long)
    test_patient_ids = torch.tensor(test_df['patient_id'].to_numpy(), dtype=torch.long)

    # Create datasets based on loss type
    if loss_type == 'contrastive':
        train_dataset = ContrastiveDataset(
            train_features,
            train_labels,
            num_pairs=EMBEDDING_TRAINING_CONFIG['num_pairs']
        )
        test_dataset = ContrastiveDataset(
            test_features,
            test_labels,
            num_pairs=1000
        )
    elif loss_type == 'triplet':
        train_dataset = TripletDataset(train_features, train_labels, train_patient_ids)
        test_dataset = TripletDataset(test_features, test_labels, test_patient_ids)
    elif loss_type == 'batch_hard_triplet':
        train_dataset = EmbeddingDataset(train_features, train_labels, train_patient_ids)
        test_dataset = EmbeddingDataset(test_features, test_labels, test_patient_ids)
    else:  # 'simple' - just features, labels, patient_ids (for classifier/evaluation)
        train_dataset = EmbeddingDataset(train_features, train_labels, train_patient_ids)
        test_dataset = EmbeddingDataset(test_features, test_labels, test_patient_ids)

    # Compute sample weights inversely proportional to patient frequency
    # This helps balance the dataset across patients
    patient_counts = train_df['patient_id'].value_counts()
    weights = train_df['patient_id'].apply(lambda x: 1.0 / patient_counts[x]).values

    return train_dataset, test_dataset, weights


# ============================================================================
# EMBEDDING TRAINING
# ============================================================================

def train_embedding_model(train_dataloader, device):
    """Train the embedding model using contrastive learning."""
    loss_type = EMBEDDING_TRAINING_CONFIG['loss_type']

    logging.info("=" * 80)
    logging.info(f"PHASE 1: Training Embedding Model with {loss_type.upper()} Loss")
    logging.info("=" * 80)

    # Initialize model
    embedding_model = DeepResidualEmbeddingModel(
        input_dim=MODEL_CONFIG['input_dim'],
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        num_blocks=MODEL_CONFIG['num_residual_blocks']
    ).to(device)

    # Select loss function based on config
    if loss_type == 'contrastive':
        loss_fn = ContrastiveLoss(temperature=EMBEDDING_TRAINING_CONFIG['temperature'])
    elif loss_type == 'triplet':
        loss_fn = TripletLoss(margin=EMBEDDING_TRAINING_CONFIG['margin'])
    else:  # batch_hard_triplet
        loss_fn = BatchHardTripletLoss(margin=EMBEDDING_TRAINING_CONFIG['margin'])

    # Optimizer
    optimizer = optim.Adam(
        embedding_model.parameters(),
        lr=EMBEDDING_TRAINING_CONFIG['learning_rate']
    )

    # Learning rate scheduler
    scheduler = None
    if EMBEDDING_TRAINING_CONFIG['use_scheduler']:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=EMBEDDING_TRAINING_CONFIG['scheduler_step_size'],
            gamma=EMBEDDING_TRAINING_CONFIG['scheduler_gamma']
        )

    # Ensure checkpoint directory exists
    os.makedirs(DATA_CONFIG['checkpoint_dir'], exist_ok=True)

    # Select training function based on loss type
    if loss_type == 'contrastive':
        embedding_model, loss_list = train_model(
            embedding_model,
            train_dataloader,
            optimizer,
            loss_fn,
            device,
            epochs=EMBEDDING_TRAINING_CONFIG['epochs'],
            scheduler=scheduler,
            checkpoint_dir=DATA_CONFIG['checkpoint_dir']
        )
    elif loss_type == 'triplet':
        embedding_model, loss_list = train_model_triplet(
            embedding_model,
            train_dataloader,
            optimizer,
            loss_fn,
            device,
            epochs=EMBEDDING_TRAINING_CONFIG['epochs'],
            scheduler=scheduler,
            checkpoint_dir=DATA_CONFIG['checkpoint_dir']
        )
    else:  # batch_hard_triplet
        embedding_model, loss_list = train_model_triplet_hard_negative(
            embedding_model,
            train_dataloader,
            optimizer,
            loss_fn,
            device,
            epochs=EMBEDDING_TRAINING_CONFIG['epochs'],
            scheduler=scheduler,
            checkpoint_dir=DATA_CONFIG['checkpoint_dir']
        )

    logging.info("Embedding model training completed!\n")
    return embedding_model


# ============================================================================
# UMAP VISUALIZATION
# ============================================================================

def generate_umap_visualizations(embedding_model, dataloader, device, results_dir, sample_size=None):
    """Generate UMAP visualizations of the learned embeddings."""
    logging.info("=" * 80)
    logging.info("Generating UMAP Visualizations")
    logging.info("=" * 80)

    embedding_model.eval()
    encoded_data = []
    labels = []
    patient_ids = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating embeddings"):
            # Handle different dataset formats
            if len(batch) == 3:  # EmbeddingDataset: (features, label, patient_id)
                features, label, patient_id = batch
            elif len(batch) == 4:  # ContrastiveDataset: (anchor, contrastive, distance, label)
                features, _, _, label = batch
                patient_id = torch.zeros_like(label)  # Dummy patient IDs
            elif len(batch) == 5:  # TripletDataset: (anchor, positive, negative, label, patient_id)
                features, _, _, label, patient_id = batch
            else:
                raise ValueError(f"Unexpected batch format with {len(batch)} elements")

            features = features.to(device)
            embeddings = embedding_model(features)
            encoded_data.extend(embeddings.cpu().numpy())
            labels.extend(label.cpu().numpy())
            patient_ids.extend(patient_id.cpu().numpy())

    # Convert to numpy arrays
    encoded_data = np.array(encoded_data)
    labels = np.array(labels)
    patient_ids = np.array(patient_ids)

    # Sample if requested
    if sample_size is not None and len(encoded_data) > sample_size:
        logging.info(f"Sampling {sample_size} examples for UMAP visualization")
        indices = np.random.choice(len(encoded_data), sample_size, replace=False)
        encoded_data = encoded_data[indices]
        labels = labels[indices]
        patient_ids = patient_ids[indices]

    logging.info(f"Generating UMAP plots with {len(encoded_data)} samples...")
    visualize_umap(encoded_data, labels, patient_ids, results_dir)
    logging.info("UMAP visualizations completed!\n")


# ============================================================================
# CLASSIFICATION
# ============================================================================

def train_and_evaluate_classifier(embedding_model, train_dataloader, test_dataloader, device, results_dir):
    """Train classifier on learned embeddings and evaluate."""
    logging.info("=" * 80)
    logging.info("PHASE 2: Training Seizure Classifier")
    logging.info("=" * 80)

    # Initialize classifier
    classifier = SeizureClassifier(
        embedding_model=embedding_model,
        embedding_dim=MODEL_CONFIG['embedding_dim'],
        num_classes=MODEL_CONFIG['num_classes']
    ).to(device)

    # Optionally unfreeze embeddings for fine-tuning
    if not CLASSIFIER_TRAINING_CONFIG['freeze_embeddings']:
        logging.info("Unfreezing embedding model for fine-tuning...")
        classifier.unfreeze_embedding_model()

    # Loss and optimizer with class weights to handle imbalance
    # Compute class weights from training data
    all_labels = []
    for batch in train_dataloader:
        # Handle different dataset formats
        if len(batch) == 3:  # EmbeddingDataset
            _, labels, _ = batch
        elif len(batch) == 4:  # ContrastiveDataset
            _, _, _, labels = batch
        elif len(batch) == 5:  # TripletDataset
            _, _, _, labels, _ = batch
        all_labels.extend(labels.numpy())

    class_counts = np.bincount(all_labels)
    class_weights = torch.tensor(1.0 / class_counts, dtype=torch.float32).to(device)
    logging.info(f"Class weights: {class_weights.cpu().numpy()}")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(
        classifier.parameters() if not CLASSIFIER_TRAINING_CONFIG['freeze_embeddings']
        else classifier.classifier.parameters(),
        lr=CLASSIFIER_TRAINING_CONFIG['learning_rate']
    )

    # Learning rate scheduler
    scheduler = None
    if CLASSIFIER_TRAINING_CONFIG['use_scheduler']:
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=CLASSIFIER_TRAINING_CONFIG['scheduler_step_size'],
            gamma=CLASSIFIER_TRAINING_CONFIG['scheduler_gamma']
        )

    # Train classifier
    classifier, train_history = train_classifier(
        classifier,
        train_dataloader,
        optimizer,
        criterion,
        device,
        epochs=CLASSIFIER_TRAINING_CONFIG['epochs'],
        scheduler=scheduler
    )

    logging.info("\nClassifier training completed!\n")

    # Plot training history
    plot_training_history(
        train_history,
        save_path=os.path.join(results_dir, 'classifier_training_history.png')
    )

    # Evaluate on train set
    train_metrics = None
    if EVAL_CONFIG['evaluate_on_train']:
        logging.info("Evaluating on training set...")
        train_metrics = evaluate_classifier(classifier, train_dataloader, device)
        print_evaluation_metrics(train_metrics, dataset_name="Train")

        # Generate visualization
        plot_evaluation_results(
            train_metrics,
            train_metrics['predictions_probs'],
            train_metrics['true_labels'],
            save_path=os.path.join(results_dir, 'train_evaluation_results.png')
        )

    # Evaluate on test set
    test_metrics = None
    if EVAL_CONFIG['evaluate_on_test']:
        logging.info("Evaluating on test set...")
        test_metrics = evaluate_classifier(classifier, test_dataloader, device)
        print_evaluation_metrics(test_metrics, dataset_name="Test")

        # Generate visualization
        plot_evaluation_results(
            test_metrics,
            test_metrics['predictions_probs'],
            test_metrics['true_labels'],
            save_path=os.path.join(results_dir, 'test_evaluation_results.png')
        )

    return classifier, train_metrics, test_metrics


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, LOGGING_CONFIG['level']),
        format=LOGGING_CONFIG['format']
    )

    # Set device
    if DEVICE_CONFIG['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{DEVICE_CONFIG['cuda_device']}")
    else:
        device = torch.device('cpu')
    logging.info(f"Using device: {device}\n")

    # Create timestamped results directory
    if DATA_CONFIG['results_dir'] is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"./results_{timestamp}"
    else:
        results_dir = DATA_CONFIG['results_dir']

    os.makedirs(results_dir, exist_ok=True)
    logging.info(f"Results will be saved to: {results_dir}\n")

    # Save configuration to JSON
    save_config_to_json(results_dir)

    # Load data for embedding training
    train_dataset_embedding, test_dataset_embedding, weights = load_train_test_dataset(
        csv_path=DATA_CONFIG['data_path'],
        loss_type=EMBEDDING_TRAINING_CONFIG['loss_type']
    )

    # Always load simple datasets for classification (regardless of embedding loss type)
    # Note: 'simple' just loads features/labels/patient_ids - no loss is applied here
    train_dataset_classifier, test_dataset_classifier, _ = load_train_test_dataset(
        csv_path=DATA_CONFIG['data_path'],
        loss_type='simple'  # Simple format: (features, labels, patient_ids)
    )

    # Create dataloaders for embedding training
    if DATA_CONFIG['use_weighted_sampling']:
        sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
        train_dataloader_embedding = DataLoader(
            train_dataset_embedding,
            batch_size=EMBEDDING_TRAINING_CONFIG['batch_size'],
            sampler=sampler
        )
    else:
        train_dataloader_embedding = DataLoader(
            train_dataset_embedding,
            batch_size=EMBEDDING_TRAINING_CONFIG['batch_size'],
            shuffle=True
        )

    # Create dataloaders for classification (always use simple format)
    train_dataloader_classifier = DataLoader(
        train_dataset_classifier,
        batch_size=CLASSIFIER_TRAINING_CONFIG['batch_size'],
        shuffle=True
    )
    test_dataloader_classifier = DataLoader(
        test_dataset_classifier,
        batch_size=CLASSIFIER_TRAINING_CONFIG['batch_size'],
        shuffle=False
    )

    # For UMAP, use the classifier dataloader (simple format)
    train_dataloader_umap = DataLoader(
        train_dataset_classifier,
        batch_size=128,
        shuffle=False
    )

    # Phase 1: Train embedding model
    embedding_model = train_embedding_model(train_dataloader_embedding, device)

    # Generate UMAP visualizations
    if EVAL_CONFIG['generate_umap']:
        generate_umap_visualizations(
            embedding_model,
            train_dataloader_umap,
            device,
            results_dir,
            sample_size=EVAL_CONFIG['umap_sample_size']
        )

    # Phase 2: Train and evaluate classifier
    classifier, train_metrics, test_metrics = train_and_evaluate_classifier(
        embedding_model,
        train_dataloader_classifier,
        test_dataloader_classifier,
        device,
        results_dir
    )

    # Final summary
    logging.info("=" * 80)
    logging.info("TRAINING COMPLETE!")
    logging.info("=" * 80)
    if test_metrics is not None:
        logging.info(f"Final Test Accuracy: {test_metrics['accuracy']:.4f}")
        logging.info(f"Final Test F1 Score: {test_metrics['f1']:.4f}")
        if test_metrics['roc_auc'] is not None:
            logging.info(f"Final Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logging.info(f"\nAll results saved to: {results_dir}")
    logging.info("Generated files:")
    logging.info("  - training_config.json")
    logging.info("  - classifier_training_history.png")
    if EVAL_CONFIG['evaluate_on_train']:
        logging.info("  - train_evaluation_results.png")
    if EVAL_CONFIG['evaluate_on_test']:
        logging.info("  - test_evaluation_results.png")
    if EVAL_CONFIG['generate_umap']:
        logging.info("  - umap_plot_labels.png")
        logging.info("  - umap_plot_patient_ids.png")


if __name__ == "__main__":
    main()
