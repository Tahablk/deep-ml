import numpy as np

def activation(x):
    '''
    Apply an activation function element-wise.
    
    Args:
        x: numpy array of any shape (raw neuron outputs)
    
    Returns:
        numpy array of same shape (activated outputs)
    
    Requirements:
        - Must be non-linear (not just returning x)
        - Must work on arrays of any shape
        - Must be deterministic
    '''
    # TODO: Implement your activation function
    a = 0.1
    b = 0.2

    negative = a * x + b * np.expm1(np.minimum(x, 0))

    result = np.where(x >= 0, x, negative)
    
    return result
