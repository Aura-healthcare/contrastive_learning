Started this work to study how we could could build an embedding which can capture the variability of seizure / non seizure but also the variability inter patients.

This is based on following papers :

SimCLR from google brain https://arxiv.org/abs/2002.05709 
MoCo from facebook ai https://arxiv.org/abs/1911.05722

Interesting paper on contrastive learning : https://arxiv.org/pdf/2010.05113

Contrastive learning in ECG or EEG :  https://pubmed.ncbi.nlm.nih.gov/37028019/

Work done on TUH dataset.
You can see an example of the training run in [this folder](results_20260214_090530) (example work).

## Usage
Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Launch training of embedding and classification head
```bash
uv run python app.py
```

## Files

### app.py
Contains the base code to launch a training

### dataset.py
Everything to build pytorch dataset which load data and create items retrieved in batch.

### loss.py
All the losses (contrastive, triplets).

### model.py
Contains the pytorch embeddings classes.

### visualization.py
Everything to create UMAPs to visualize embedding space in 2D.


## What I did

### Step 1
Simple depth model, trial of contrastive loss and triplet loss
Not converging, UMAP and embedding PCA shows mixed seizure/non seizure

### Step 2
Denser model with triplet loss. Need to have residual connexion to avoid vanishing gradients.
But the triplets are not well chosen (not balanced, and no hard enough).

Trial of projection head to reduce the model only during training. Not good for now.

### Step 3
Work on hard tiplets. If the triplets are not hard enough, the model cannot learn.
With BatchHardTripletLoss, triplets are not done in the Dataset, but chosen during the loss computation.

### Observation

Training data need to be balanced (too many non seizure).

TODO: try embedding only with patients to capture patients differences.

This is unsupervised in terms of class (e.g., seizure/no seizure), but weakly supervised using patient identity.
I also increased the learning rate

/!\ It seems that with my implementation, loss cannot go under 1 -> depends on marging

Contrastive works well for separating 3 patients with margin = 1, but not so well on labels 1 or 0
Lowering margin to 0.5 gives an embedding that separates better the labels than the patients.

UMAP on 25 patients with plotly shows complex problem.
Need to do umap for each patient.

### Things to experiment
Multi-stage transfer learning approach:                                                                                
  1. Phase 1: Learn rich patient embeddings (discriminate between patients)                                                                       
  2. Phase 2: Fine-tune for seizure detection (unfreeze some layers)