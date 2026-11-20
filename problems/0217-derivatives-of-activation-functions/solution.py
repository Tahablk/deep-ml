import math

def activation_derivatives(x: float) -> dict[str, float]:

    # Sigmoid
    if x >= 0:
        sigmoid = 1 / (1 + math.exp(-x))
    else:
        sigmoid = math.exp(x) / (1 + math.exp(x))

    sigmoid_derivative = sigmoid * (1 - sigmoid)

    # Tanh
    tanh_derivative = 1 - math.tanh(x)**2

    # ReLU
    relu_derivative = 1.0 if x > 0 else 0.0

    return {
        "sigmoid": sigmoid_derivative,
        "tanh": tanh_derivative,
        "relu": relu_derivative
    }