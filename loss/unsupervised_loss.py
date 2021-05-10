import torch

"""
This file contains all loss functions that are "unsupervised" - i.e., they will take in only one tensor ideally,
which should be a tensor containing n image samples. See specific examples and docstrings.
"""


def l1_loss(results):
    """
    Basic temporal loss. Computes L1 distance between result[i] and result[i+1].
    """
    losses = results[1:] - results[:-1]

    return losses.sum()

