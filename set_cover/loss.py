import torch


def kl_loss(mu, log_sigma):
    return -0.5 * torch.mean(
            torch.sum(1 + 2 * log_sigma - mu**2 - log_sigma.exp()**2, dim=1))