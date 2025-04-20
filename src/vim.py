from sklearn.decomposition import PCA
import torch
import numpy as np
import torch.nn.functional as F


def compute_principal_subspace(features, n_components):
    pca = PCA(n_components=n_components)
    pca.fit(features)
    return pca

def compute_virtual_logit(feature, pca, alpha):
    projected = pca.inverse_transform(pca.transform(feature.reshape(1, -1)))
    residual = feature - projected.flatten()
    virtual_logit = alpha * np.linalg.norm(residual)
    return virtual_logit

def compute_ood_score(logits, virtual_logit):
    augmented_logits = torch.cat([logits, torch.tensor([virtual_logit])])
    probabilities = F.softmax(augmented_logits, dim=0)
    ood_score = probabilities[-1].item()
    return ood_score