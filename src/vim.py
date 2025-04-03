from sklearn.decomposition import PCA
import numpy as np
import torch.nn.functional as F


def compute_principal_subspace(features, n_components):
    """
    Compute the principal subspace using PCA.
    Args:
        features (numpy.ndarray): Feature matrix of shape (n_samples, feature_dim).
        n_components (int): Number of principal components to retain.
    Returns:
        pca (PCA object): Trained PCA model.
    """
    pca = PCA(n_components=n_components)
    pca.fit(features)
    return pca

def compute_virtual_logit(feature, pca, alpha):
    """
    Compute the virtual logit for a given feature vector.
    Args:
        feature (numpy.ndarray): Feature vector of shape (feature_dim,).
        pca (PCA object): Trained PCA model.
        alpha (float): Scaling factor for the virtual logit.
    Returns:
        virtual_logit (float): Computed virtual logit.
    """
    # Project feature onto principal subspace
    projected = pca.inverse_transform(pca.transform(feature.reshape(1, -1)))
    residual = feature - projected.flatten()
    virtual_logit = alpha * np.linalg.norm(residual)
    return virtual_logit

def compute_ood_score(logits, virtual_logit):
    """
    Compute the OOD score by augmenting logits with the virtual logit.
    Args:
        logits (torch.Tensor): Original logits of shape (num_classes,).
        virtual_logit (float): Computed virtual logit.
    Returns:
        ood_score (float): Probability corresponding to the virtual logit.
    """
    # Append virtual logit to original logits
    augmented_logits = torch.cat([logits, torch.tensor([virtual_logit])])
    probabilities = F.softmax(augmented_logits, dim=0)
    ood_score = probabilities[-1].item()
    return ood_score