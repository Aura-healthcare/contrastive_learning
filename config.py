"""
Configuration file for contrastive learning seizure detection pipeline.

Supports two dataset formats:
  - 'tuh': TUH ECG features (14 HRV features, columns: patient_id, split, label)
  - 'teppe': Teppe dataset (29 HRV + 141 ACC features, columns: patient-id, training-split, label)

Set DATASET_FORMAT below to switch between them.
"""

# ============================================================================
# DATASET FORMAT — set this to 'tuh' or 'teppe'
# ============================================================================
DATASET_FORMAT = 'teppe'

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

# --- TUH format ---
_TUH_DATA_CONFIG = {
    'data_path': '/Users/laura/Documents/aura/tuh_ecg_features2.csv',
    'patient_id_col': 'patient_id',
    'label_col': 'label',
    'split_col': 'split',
    'train_split_name': 'train',
    'test_split_name': 'dev',
    'drop_columns': ['interval_index', 'interval_start_time', 'montage', 'session_id', 'file_id'],
    'availability_cols': [],  # TUH has no availability flags
}

# --- Teppe format ---
_TEPPE_DATA_CONFIG = {
    'data_path': '/home/adm-laura/git/feat-grid-all-intersect-clean.csv',
    'patient_id_col': 'patient-id',
    'label_col': 'label',
    'split_col': 'training-split',
    'train_split_name': 'train',
    'test_split_name': 'test',
    'drop_columns': ['timestamps', 'hrv_interval_index', 'hrv_interval_start_time'],
    'availability_cols': ['hrv_available', 'acc_available'],  # sensor availability flags
}

# Merge with shared settings
_SHARED_DATA_CONFIG = {
    'checkpoint_dir': './checkpoints/',
    'results_dir': None,  # Auto-generated with timestamp if None
    'num_patients_train': None,  # Limit training patients (None for all)
    'use_weighted_sampling': True,
    'per_patient_normalization': True,  # z-score features relative to each patient's baseline
}

DATA_CONFIG = {**_SHARED_DATA_CONFIG, **(_TEPPE_DATA_CONFIG if DATASET_FORMAT == 'teppe' else _TUH_DATA_CONFIG)}


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

# input_dim is computed automatically from the data in launch_train.py
# Set to None for auto-detection, or override with an int.
MODEL_CONFIG = {
    'input_dim': None,  # Auto-detected from data
    'embedding_dim': 64,  # Reduced from 128 (better for generalisation)
    'num_classes': 2,

    'num_residual_blocks': 2,  # Reduced from 3
    'embedding_dropout': 0.4, # Increased from 0.2 to reduce overfitting on seizure samples

    'classifier_hidden_dims': [64, 32],
    'classifier_dropout': 0.3,
}


# ============================================================================
# TRAINING CONFIGURATION - EMBEDDING MODEL
# ============================================================================

EMBEDDING_TRAINING_CONFIG = {
    'epochs': 30,   # Reduced from 100 — fewer seizure samples get over-compressed with too many epochs
    'batch_size': 1024,
    'learning_rate': 1e-3,  # Reduced from 5e-3

    # Loss function options: 'contrastive', 'triplet', 'batch_hard_triplet', 'supcon'
    'loss_type': 'supcon',

    # Loss function parameters
    'margin': 1.0,              # For triplet and batch_hard_triplet
    'temperature': 0.07,        # Reduced from 0.3 — sharper focus on hard negatives
    'num_pairs': 5000,          # For contrastive dataset
    'pk_p': 16,                 # PKBatchSampler: number of patients per batch (supcon)
    'pk_k': 16,                 # PKBatchSampler: number of samples per patient per batch (supcon)
    'max_per_patient': 4,       # Cap per-patient samples per class per batch (forces cross-patient diversity)

    # Learning rate scheduler
    'use_scheduler': True,
    'scheduler_step_size': 10,
    'scheduler_gamma': 0.5,

    # Checkpointing
    'save_checkpoints': True,
    'checkpoint_frequency': 1,  # Save every N epochs
}


# ============================================================================
# TRAINING CONFIGURATION - CLASSIFIER
# ============================================================================

CLASSIFIER_TRAINING_CONFIG = {
    'epochs': 5,  # Set low for testing (use 20+ for real training)
    'batch_size': 128,
    'learning_rate': 1e-4,

    # Learning rate scheduler
    'use_scheduler': True,
    'scheduler_step_size': 5,
    'scheduler_gamma': 0.5,

    # Fine-tuning options (Two-stage approach)
    'freeze_embeddings': True,  # Changed: freeze embeddings by default
    'unfreeze_last_n_blocks': None,
}


# ============================================================================
# EVALUATION CONFIGURATION
# ============================================================================

EVAL_CONFIG = {
    'evaluate_on_train': True,  # Whether to evaluate on training set
    'evaluate_on_test': True,   # Whether to evaluate on test set
    'generate_umap': True,      # Whether to generate UMAP visualizations
    'umap_sample_size': 2000,   # Number of samples to use for UMAP (None for all)
}


# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

DEVICE_CONFIG = {
    'use_cuda': False,  # Set to False to force CPU
    'cuda_device': 0,  # GPU device number
}


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'format': '%(asctime)s - %(levelname)s - %(message)s',
}
