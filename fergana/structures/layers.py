import fergana.structures.neurons as neurons
from fergana.math import functions
from abc import ABC, abstractmethod
import random


class __Layer(ABC):
    @abstractmethod
    def __init__(self, number_of_neurons: int):
        pass

    @abstractmethod
    def get_values(self):
        pass

    @abstractmethod
    def clear_data(self):
        pass

    @abstractmethod
    def __copy__(self):
        pass


class InputLayer(__Layer):
    def __init__(self, number_of_neurons: int):
        self.neurons = [neurons.InputNeuron() for neuron in range(number_of_neurons)]

    def get_values(self):
        return [neuron.value for neuron in self.neurons]

    def clear_data(self):
        for neuron in self.neurons:
            neuron.value = None

    def __copy__(self):
        new_layer = InputLayer(len(self.neurons))
        return new_layer


class HiddenLayer(__Layer):
    def __init__(self, number_of_neurons: int, activation_function=functions.sigmoid):
        self.neurons = [neurons.HiddenNeuron(activation_function) for neuron in range(number_of_neurons)]
        self.linked_layer = None

    def __copy__(self):
        new_layer = HiddenLayer(number_of_neurons=len(self.neurons),
                                activation_function=self.neurons[0].activation_function)
        return new_layer

    def get_values(self):
        self.calculate_layer()
        return [neuron.value for neuron in self.neurons]

    def clear_data(self):
        for neuron in self.neurons:
            neuron.value = None
            neuron.gradient = None

    def set_links(self):
        """
        define links between neighbor layers neurons
        """
        for neuron in self.neurons:
            neuron.links[1] = self.linked_layer.neurons
            neuron.links[0] = [random.uniform(-1, 1) for linked_neuron in neuron.links[1]]
            for linked_neuron in neuron.links[1]:
                linked_neuron.linked_neurons.append(neuron)

    def calculate_layer(self):
        """
        update values of layer neurons
        """
        for neuron in self.neurons:
            neuron.update_value()

    def get_gradients(self) -> list[float]:
        """
        :return: list of layer neurons gradients
        """
        return [neuron.gradient for neuron in self.neurons]

    def get_link_weights(self) -> list[list[float]]:
        """
        :return: list of layer neurons individual weights values
        """
        return [neuron.links[0] for neuron in self.neurons]


class OutputLayer(__Layer):
    def __init__(self, number_of_neurons: int, activation_function=functions.sigmoid):
        self.neurons = [neurons.OutputNeuron(activation_function) for neuron in range(number_of_neurons)]
        self.linked_layer = None

    def __copy__(self):
        new_layer = OutputLayer(number_of_neurons=len(self.neurons),
                                activation_function=self.neurons[0].activation_function)
        return new_layer

    def get_values(self):
        self.calculate_layer()
        return [neuron.value for neuron in self.neurons]

    def clear_data(self):
        for neuron in self.neurons:
            neuron.value = None
            neuron.gradient = None

    def set_links(self):
        """
        define links between neighbor layers neurons
        """
        for neuron in self.neurons:
            neuron.links[1] = self.linked_layer.neurons
            neuron.links[0] = [random.uniform(-1, 1) for linked_neuron in neuron.links[1]]
            for linked_neuron in neuron.links[1]:
                linked_neuron.linked_neurons.append(neuron)

    def calculate_layer(self):
        """
        update values of layer neurons
        """
        for neuron in self.neurons:
            neuron.update_value()

    def get_gradients(self) -> list[float]:
        """
        :return: list of layer neurons gradients
        """
        return [neuron.gradient for neuron in self.neurons]

    def get_link_weights(self) -> list[list[float]]:
        """
        :return: list of layer neurons individual weights values
        """
        return [neuron.links[0] for neuron in self.neurons]
