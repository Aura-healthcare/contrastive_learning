Step 1 :
Simple depth model, trial of contrastive loss and triplet loss
Not converging, UMAP and embedding PCA shows mixed seizure/non seizure

Step 2 :
Denser model with triplet loss. Need to have residual connexion to avoid vanishing gradients.
But the triplets are not well chosen (not balanced, and no hard enough).

Trial of projection head to reduce the model only during training. Not good for now.

Step 3 :
Work on hard tiplets. If the triplets are not hard enough, the model cannot learn.
With BatchHardTripletLoss, triplets are not done in the Dataset, but chosen during the loss computation.

Training data need to be balanced (too many non seizure).

TODO: try embedding only with patients to capture patients differences.
This is unsupervised in terms of class (e.g., seizure/no seizure), but weakly supervised using patient identity.
I also increased the learning rate
/!\ It seems that with my implementation, loss cannot go under 1 -> depends on marging
Contrastive works well for separating 3 patients with margin = 1, but not so well on labels 1 or 0
Lowering margin to 0.5 gives an embedding that separates better the labels than the patients.

UMAP on 25 patients with plotly shows complex problem.
Need to do umap for each patient.