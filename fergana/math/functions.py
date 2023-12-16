import math
import sys


# block of loss functions
def true_loss(model_out: list[float], true_out: list[float]) -> list[float]:
    return [neuron_out - true_out for neuron_out, true_out in zip(model_out, true_out)]


def num_into_interval(number: float, interval: list) -> float:
    return min(max(number, interval[0]), interval[1])


def softmax(values: list) -> list[float]:
    return list(map(lambda x: (math.e ** x) / sum([math.e ** values[j] for j in range(len(values))]), values))


def sigmoid(x) -> float:
    return 1 / (1 + math.exp(x * -1))


def derivative_sigmoid(x) -> float:
    return x * (1 - x)


def relu(x) -> float:
    return max(0, x)


def derivative_relu(x) -> float:
    return 1 if x >= 0 else 0


def tanh(x) -> float:
    return (2 / (1 + math.exp(x * -2))) - 1


def derivative_tanh(x) -> float:
    return 1 - x ** 2


def softplus(x) -> float:
    return math.log(1 + math.exp(x))


def derivative_softplus(x) -> float:
    return 1 / (1 + math.exp(x * -1))


def linear(x) -> float:
    return x


def derivative_linear(x) -> float:
    return 1


def str_to_function(class_string: str):
    return getattr(sys.modules[__name__], class_string)


def str_to_class(class_string: str):
    return getattr(sys.modules[__name__], class_string)


pairs = {
    sigmoid: derivative_sigmoid,
    relu: derivative_relu,
    tanh: derivative_tanh,
    softplus: derivative_softplus,
    linear: derivative_linear
}
