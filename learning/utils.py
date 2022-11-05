import numpy as np

def compute_softmax_prob(actor_w, tiles):
    """
    Computes softmax probability for all actions
    
    Args:
    actor_w - np.array, an array of actor weights
    tiles - np.array, an array of active tiles
    
    Returns:
    softmax_prob - np.array, an array of size equal to num. actions, and sums to 1.
    """
    state_action_preferences = []
    state_action_preferences = actor_w[:, tiles].sum(axis=1)
    c = np.max(state_action_preferences)
    numerator = np.exp(state_action_preferences - c)
    denominator = np.sum(numerator)
    softmax_prob = numerator / denominator
    return softmax_prob