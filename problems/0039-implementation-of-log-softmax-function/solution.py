import numpy as np

def log_softmax(scores: np.ndarray) -> np.ndarray:
    # Your code here
    max_score = max(scores)
    scores = [np.exp(x - max_score) for x in scores]
    total = sum(scores)
    scores = [np.log(x / total) for x in scores] 
    return scores                            
    
    pass