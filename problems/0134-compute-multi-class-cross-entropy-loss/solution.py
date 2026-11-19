import numpy as np

def compute_cross_entropy_loss(
    predicted_probs: np.ndarray, 
    true_labels: np.ndarray, 
    epsilon = 1e-15
    ) -> float:
     # Avoid log(0)
    probs = np.clip(predicted_probs, epsilon, 1 - epsilon)

    # One-hot encoded labels
    if true_labels.shape == probs.shape:
        losses = -np.sum(true_labels * np.log(probs), axis=-1)

    # Integer class labels
    else:
        losses = -np.log(
            probs[np.arange(len(true_labels)), true_labels]
        )

    return float(np.mean(losses))
    
    