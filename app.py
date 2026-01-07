import pandas as pd
import logging
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler, Dataset
import torch.optim as optim
from model import EmbeddingModel, DeepResidualEmbeddingModel
from loss import ContrastiveLoss, TripletLoss, BatchHardTripletLoss
import os
from dataset import ContrastiveDataset, TripletDataset, EmbeddingDataset
from train import train_model, train_model_triplet, compute_distances, plot_distance_distributions, train_model_triplet_hard_negative
import numpy as np
from functools import lru_cache

@lru_cache(maxsize=1)
def load_train_test_dataset(csv_path) -> tuple[Dataset, Dataset, list[float]]:
    csv_path = '/Users/laura/Documents/aura/tuh_ecg_features2.csv'  
    df = pd.read_csv(csv_path)

    df.drop(columns=['interval_index', 'interval_start_time', 'montage', 'session_id', 'file_id'], inplace=True)

    # Normalize data and convert to tensor
    train_df = df[df['split'] == 'train'].drop(columns=['split'])
    test_df = df[df['split'] == 'dev'].drop(columns=['split'])

    # Remove 25% of majority class (label 0) randomly
    # label_0_mask = train_df['label'] == 0
    # remove_mask = label_0_mask & (np.random.random(len(train_df)) < 0.8)
    # train_df = train_df[~remove_mask]

    # Only keep 3 patients in train
    patient_ids_list = train_df['patient_id'].unique()
    train_df = train_df[train_df['patient_id'].isin(patient_ids_list[:10])]

    # Normalize features
    feature_columns = train_df.drop(columns=['label', 'patient_id']).columns
    train_mean = train_df[feature_columns].mean()
    train_std = train_df[feature_columns].std()

    train_df[feature_columns] = (train_df[feature_columns] - train_mean) / train_std
    test_df[feature_columns] = (test_df[feature_columns] - train_mean) / train_std

    # Create DataLoader
    train_features = torch.tensor(train_df.drop(columns=['label', 'patient_id']).to_numpy(), dtype=torch.float32)
    train_labels = torch.tensor(train_df['label'].to_numpy(), dtype=torch.long)
    test_features = torch.tensor(test_df.drop(columns=['label', 'patient_id']).to_numpy(), dtype=torch.float32)
    test_labels = torch.tensor(test_df['label'].to_numpy(), dtype=torch.long)

    train_patient_ids = torch.tensor(train_df['patient_id'].to_numpy(), dtype=torch.long)
    test_patient_ids = torch.tensor(test_df['patient_id'].to_numpy(), dtype=torch.long)

    # train_dataset = ContrastiveDataset(train_features, train_labels, num_pairs=5000)
    # test_dataset = ContrastiveDataset(test_features, test_labels, num_pairs=1000)

    # train_dataset = TripletDataset(train_features, train_labels, train_patient_ids)
    # test_dataset = TripletDataset(test_features, test_labels, test_patient_ids)
    train_dataset = EmbeddingDataset(train_features, train_labels, train_patient_ids)
    test_dataset = EmbeddingDataset(test_features, test_labels, test_patient_ids)

    # Compute sample weights inversely proportional to patient frequency to balance the dataset
    patient_counts = train_df['patient_id'].value_counts()
    weights = train_df['patient_id'].apply(lambda x: 1.0 / patient_counts[x]).values

    return train_dataset, test_dataset, weights
    
    
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    train_dataset, test_dataset, weights = load_train_test_dataset(csv_path="/Users/laura/Documents/aura/tuh_ecg_features2.csv")

    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_dataloader = DataLoader(train_dataset, batch_size=1024, sampler=sampler)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True)

    net = DeepResidualEmbeddingModel(input_dim=14, embedding_dim=1024)

    device = 'cpu'
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        
    net = net.to(device)

    # Define the training configuration
    # loss_fn = TripletLoss(margin=2.0)
    loss_fn = BatchHardTripletLoss(margin=1.0)
    optimizer = optim.Adam(net.parameters(), lr=5e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.3)

    # Define a directory to save the checkpoints
    checkpoint_dir = './checkpoints/'

    # Ensure the directory exists
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # train_model(
    #     net,
    #     train_dataloader,
    #     optimizer,
    #     loss_fn,
    #     device,
    #     epochs=30,
    #     scheduler=scheduler,
    #     checkpoint_dir=checkpoint_dir
    # )

    # train_model_triplet(
    #     net,
    #     train_dataloader,
    #     optimizer,
    #     loss_fn,
    #     device,
    #     epochs=30,
    #     scheduler=scheduler,
    #     checkpoint_dir=checkpoint_dir
    # )

    train_model_triplet_hard_negative(
        net,
        train_dataloader,
        optimizer,
        loss_fn,
        device,
        epochs=30,
        scheduler=scheduler,
        checkpoint_dir=checkpoint_dir
    )

    # pos_dists, neg_dists = compute_distances(
    #     net,
    #     train_dataloader,
    #     device,
    #     k=100
    # )

    # plot_distance_distributions(pos_dists, neg_dists)


