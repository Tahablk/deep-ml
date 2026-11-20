import math 
def softmax_derivative(x: list[float]) -> list[list[float]]:
    """
    Compute the Jacobian matrix of the softmax function.
    
    Args:
        x: Input vector of real numbers
        
    Returns:
        Jacobian matrix J where J[i][j] = d(softmax_i)/d(x_j)
    """
    # Your code here
    softmax_sum = 0
    softmax_max = max(x)

    for e in x:
        softmax_sum += math.exp(e-softmax_max)

    softmax_list = [
        math.exp(e-softmax_max) / softmax_sum for e in x]

    n = len(x)
    jacobian = [[0.0] * n for _ in range(n)]

    for i in range(n):
        for j in range(n):
            if i ==j:  
                jacobian[i][j] = softmax_list[i] * (1 - softmax_list[i])
            else :
                jacobian[i][j] = (-softmax_list[i]) * softmax_list[j]
    return jacobian

