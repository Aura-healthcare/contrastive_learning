## Plan

1. Pretrain with contrastive learning
2. Fine-tune with: (partial fine tuning since not a lot of datasets or full fine tuning with low learning rate)
	- classification loss
	- contrastive loss (weighted) : L=Lclassification​+λ⋅Lcontrastive​ (λ = 0.1 or 0.2)
3. Evaluate with:
	- kNN : train embedding + classify using k-nearest neighbors in embedding space
	- linear probe : train embedding + freeze and train logistic regression classifier
	- LOPO classifier
	- retrieval metrics : Precision@K, Recall@K, Mean Average Precision (mAP) -> using cosine similarity for distance
4. Visualize embedding:
	- color by patient vs label



## Training strategy:
1. training embedding + total freeze and train classifier
2. training embedding and classifier at same time -> hard
3. training embedding + fine tune with classifier -> best approach


## Positive / Negative pair choice:
Focus first on good contrastive loss (then after triplet, more complex)
We want invariance to:
- patient identity
- baseline cardiac differences
- sensor noise

So positives could be:
- same seizure segment under augmentations
- nearby windows within the same seizure
- maybe similar physiological states

Negatives:
- seizure vs non-seizure (obvious)
- BUT also: non-seizure from different contexts
Pitfall:If we define positives only within-patient, we risk learning patient-specific embeddings, which is exactly what we don’t want.

Consider explicitly encouraging:
- cross-patient positives for seizures
- or patient-adversarial training