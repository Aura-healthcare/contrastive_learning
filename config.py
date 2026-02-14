"""
Configuration file for contrastive learning seizure detection pipeline.
"""

# ============================================================================
# DATA CONFIGURATION
# ============================================================================

DATA_CONFIG = {
    # Data paths
    'data_path': '/Users/laura/Documents/aura/tuh_ecg_features2.csv',
    'checkpoint_dir': './checkpoints/',
    'results_dir': None,  # Will be auto-generated with timestamp if None

    # Data preprocessing
    'num_patients_train': 3,  # Limit training to N patients for quick experiments (None for all)
    'use_weighted_sampling': True,  # Use weighted sampling to balance across patients

    # Features to drop
    'drop_columns': ['interval_index', 'interval_start_time', 'montage', 'session_id', 'file_id'],

    # Data splits
    'train_split_name': 'train',
    'test_split_name': 'dev',
}


# ============================================================================
# MODEL CONFIGURATION
# ============================================================================

MODEL_CONFIG = {
    # Input/output dimensions
    'input_dim': 14,
    'embedding_dim': 1024,
    'num_classes': 2,

    # Embedding model architecture
    'num_residual_blocks': 3,  # Number of residual blocks in each stage

    # Classifier architecture
    'classifier_hidden_dims': [256, 64],  # Hidden layer sizes in classifier head
    'classifier_dropout': 0.3,
}


# ============================================================================
# TRAINING CONFIGURATION - EMBEDDING MODEL
# ============================================================================

EMBEDDING_TRAINING_CONFIG = {
    'epochs': 10,
    'batch_size': 1024,
    'learning_rate': 5e-3,

    # Loss function options: 'contrastive', 'triplet', 'batch_hard_triplet'
    'loss_type': 'batch_hard_triplet',

    # Loss function parameters
    'margin': 1.0,              # For triplet and batch_hard_triplet
    'temperature': 0.1,         # For contrastive loss
    'num_pairs': 5000,          # For contrastive dataset

    # Learning rate scheduler
    'use_scheduler': True,
    'scheduler_step_size': 7,
    'scheduler_gamma': 0.3,

    # Checkpointing
    'save_checkpoints': True,
    'checkpoint_frequency': 1,  # Save every N epochs
}


# ============================================================================
# TRAINING CONFIGURATION - CLASSIFIER
# ============================================================================

CLASSIFIER_TRAINING_CONFIG = {
    'epochs': 20,
    'batch_size': 128,
    'learning_rate': 1e-3,

    # Learning rate scheduler
    'use_scheduler': True,
    'scheduler_step_size': 5,
    'scheduler_gamma': 0.5,

    # Fine-tuning options
    'freeze_embeddings': True,  # If False, will fine-tune embedding model during classifier training
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
    'use_cuda': True,  # Set to False to force CPU
    'cuda_device': 0,  # GPU device number
}


# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING_CONFIG = {
    'level': 'INFO',  # 'DEBUG', 'INFO', 'WARNING', 'ERROR'
    'format': '%(asctime)s - %(levelname)s - %(message)s',
}
