import torch
from torch.utils.data import Dataset
import numpy as np
import random
from collections import defaultdict

class ContrastiveDataset(Dataset):
    def __init__(self, features, labels, num_pairs=1000):
        self.features = features
        self.labels = labels
        self.num_pairs = num_pairs
        
        # Create label indices for better sampling
        self.label_to_indices = {}
        for idx, label in enumerate(labels):
            if label.item() not in self.label_to_indices:
                self.label_to_indices[label.item()] = []
            self.label_to_indices[label.item()].append(idx)
        
    def __len__(self):
        return self.num_pairs
    
    def __getitem__(self, idx):
        # 50% chance of positive pair (same label), 50% chance of negative pair (different label)
        if np.random.random() < 0.5:
            # Positive pair - same label
            label = np.random.choice(list(self.label_to_indices.keys()))
            if len(self.label_to_indices[label]) >= 2:
                idx1, idx2 = np.random.choice(self.label_to_indices[label], size=2, replace=False)
                distance = 0.0
            else:
                # Fallback to random if not enough samples
                idx1 = np.random.randint(0, len(self.features))
                idx2 = np.random.randint(0, len(self.features))
                distance = 0.0 if self.labels[idx1] == self.labels[idx2] else 1.0
        else:
            # Negative pair - different labels
            labels = list(self.label_to_indices.keys())
            if len(labels) >= 2:
                label1, label2 = np.random.choice(labels, size=2, replace=False)
                idx1 = np.random.choice(self.label_to_indices[label1])
                idx2 = np.random.choice(self.label_to_indices[label2])
                distance = 1.0
            else:
                # Fallback to random
                idx1 = np.random.randint(0, len(self.features))
                idx2 = np.random.randint(0, len(self.features))
                distance = 0.0 if self.labels[idx1] == self.labels[idx2] else 1.0
        
        anchor = self.features[idx1]
        contrastive = self.features[idx2]
        anchor_label = self.labels[idx1]
        
        return anchor, contrastive, torch.tensor(distance, dtype=torch.float32), anchor_label

class TripletDataset(Dataset):
    def __init__(self, features, labels, patient_ids):
        self.features = features
        self.labels = labels
        self.patient_ids = patient_ids
        
        self.label_patient_index = self._build_index()
        self.triplets = self._create_triplets()
        
    def _build_index(self):
        index = defaultdict(list)
        for i, (lbl, pid) in enumerate(zip(self.labels.tolist(), self.patient_ids.tolist())):
            index[(lbl, pid)].append(i)
        return index

    def _create_triplets(self):
        triplets = []
        # Go through all label-patient groups to generate triplets
        for (label, patient), anchor_pool in self.label_patient_index.items():
            if len(anchor_pool) < 2:
                # Need at least 2 samples for positive pairs
                continue
            
            # Negative candidates are all indices with different label
            negative_candidates = []
            for (neg_label, neg_patient), neg_pool in self.label_patient_index.items():
                if neg_label != label:
                    negative_candidates.extend(neg_pool)
            
            if not negative_candidates:
                # No negatives found for this label
                continue
            
            # For each anchor in this group, create triplets
            for anchor_idx in anchor_pool:
                # Positive is any different sample with same label and patient
                positive_pool = [i for i in anchor_pool if i != anchor_idx]
                for positive_idx in positive_pool:
                    # For each positive, randomly sample one negative
                    negative_idx = random.choice(negative_candidates)
                    triplets.append((anchor_idx, positive_idx, negative_idx))
        
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        anchor_idx, positive_idx, negative_idx = self.triplets[idx]
        return (
            self.features[anchor_idx],
            self.features[positive_idx],
            self.features[negative_idx],
            self.labels[anchor_idx],
            self.patient_ids[anchor_idx],
        )

class EmbeddingDataset(Dataset):
    def __init__(self, features, labels, patient_ids):
        self.features = features  # Tensor or array of shape (N, D)
        self.labels = labels      # Tensor or array of shape (N,)
        self.patient_ids = patient_ids  # Tensor or array of shape (N,)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return (
            self.features[idx],
            self.labels[idx],
            self.patient_ids[idx],
        )