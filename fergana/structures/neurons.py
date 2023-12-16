from abc import ABC, abstractmethod
import random
from fergana.math import functions


class __Neuron(ABC):

    def __init__(self):
        self.value = None

    @abstractmethod
    def __copy__(self):
        pass


class InputNeuron(__Neuron):
    def __init__(self):
        super().__init__()
        self.linked_neurons = []

    def __copy__(self):
        new_neuron = InputNeuron()
        return new_neuron


class HiddenNeuron(__Neuron):
    def __init__(self, activation_function):
        super().__init__()
        self.activation_function = activation_function
        self.derivative_function = functions.pairs[activation_function]
        self.links = [[], []]
        self.linked_neurons = []
        self.bias_weight = random.random()
        self.gradient = None

    def __copy__(self):
        new_neuron = HiddenNeuron(activation_function=self.activation_function)
        return new_neuron

    def update_value(self):
        """
        update neuron value
        """
        self.value = self.activation_function(sum(
            [neuron.value * weight if neuron.value is not None else 0 for neuron, weight in
             zip(self.links[1],
                 self.links[0])]) + 1 * self.bias_weight)

    def update_gradient(self, local_error: float):
        self.gradient = local_error * self.derivative_function(self.value)

    def create_link(self, linked_neuron):
        """
        create a link between 2 neurons
        :param linked_neuron: neuron to which a link will be created
        """
        self.links[0].append(random.uniform(0, 1))
        self.links[1].append(linked_neuron)
        linked_neuron.linked_neurons.append(self)


class OutputNeuron(__Neuron):
    def __init__(self, activation_function):
        super().__init__()
        self.activation_function = activation_function
        self.derivative_function = functions.pairs[activation_function]
        self.links = [[], []]
        self.bias_weight = random.random()
        self.gradient = None

    def __copy__(self):
        new_neuron = OutputNeuron(activation_function=self.activation_function)
        return new_neuron

    def update_value(self):
        """
        update neuron value
        """
        self.value = self.activation_function(sum(
            [neuron.value * weight if neuron.value is not None else 0 for neuron, weight in
             zip(self.links[1],
                 self.links[0])]) + 1 * self.bias_weight)

    def create_link(self, linked_neuron):
        """
        create a link between 2 neurons
        :param linked_neuron: neuron to which a link will be created
        """
        self.links[0].append(random.uniform(0, 1))
        self.links[1].append(linked_neuron)
        linked_neuron.linked_neurons.append(self)

    def update_gradient(self, local_error: float):
        self.gradient = local_error * self.derivative_function(self.value)
