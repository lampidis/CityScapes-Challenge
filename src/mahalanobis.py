import torch
from scipy.spatial import distance
import numpy as np

def batch_distribution(x):
    # x torch Shape: [Batch, Features, Height, Weight]
    x = x.permute(0, 2, 3, 1)
    x = x.reshape(-1, x.shape[-1])
    mean = x.mean(dim=0)  # Shape: [Features]
    x_centered = x - mean
    n_samples = x_centered.shape[0]
    cov = (x_centered.T @ x_centered) / (n_samples - 1)  # Shape: [Features, Features]

    return mean, cov

# Update Mean 
# \[
# \mu_{\text{new}} = \frac{n \mu + n_b \mu_b}{n + n_b}
# \]
# Update Covariance
# \[
# \Sigma_{\text{new}} = \frac{n \Sigma + n_b \Sigma_b + \frac{n n_b}{n + n_b} (\mu - \mu_b)(\mu - \mu_b)^T}{n + n_b}
# \]
def update_global_distribution(global_mean, global_cov, features, itr):
    [B,F,H,W] = features.shape
    b_mean, b_cov = batch_distribution(features)

    global_samples = B*itr
    total_samples = global_samples+B

    delta_mean = global_mean - b_mean
    new_mean = (global_samples * global_mean + B * b_mean) / total_samples

    mean_diff = delta_mean.reshape(-1, 1)
    cov_adjustment = (global_samples * B / total_samples) * (mean_diff @ mean_diff.T)

    new_cov = (B*b_cov + global_samples*global_cov + cov_adjustment) / total_samples

    return new_mean, new_cov


def mahalanobis_distance(x, mean, cov):
    x = x.view(x.size(0), -1)
    x_mean = x.mean(dim=1)
    
    centered_x = x_mean - mean
    # print("Condition number:", np.linalg.cond(cov))
    cov_inv = torch.linalg.pinv(cov)
    diff = centered_x - mean
    mahalanobis = torch.sqrt(diff @ cov_inv @ diff.T)
    # print(mahalanobis)
    return mahalanobis