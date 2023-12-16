from abc import ABC, abstractmethod
from fergana.math import functions
import json


class __Model(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def __copy__(self):
        """
        Make copy of model, with all same parameters
        """
        pass

    @abstractmethod
    def compile(self):
        """
        Define all links between neurons and make layer structure immutable
        """
        pass

    @abstractmethod
    def clear_values(self):
        """
        clear all values in model, like gradients of neurons, values of neurons and so on
        """
        pass

    @abstractmethod
    def data_load(self, data: list, number_of_layer=0):
        """
        load values of neurons to some layer of model, by default for 0 layer
        :param data: values which will be loaded
        :param number_of_layer: index of layer which will accept data
        """
        pass

    @abstractmethod
    def accuracy_estimate(self, estimate_set: list[list[list, list]], loss_function=functions.true_loss):
        pass

    @abstractmethod
    def save_model(self, name_of_model: str):
        pass

    @staticmethod
    def load_model(file_source: str):
        pass


class PerceptronModel(__Model):
    def __init__(self, *layers):
        self.neural_network = list(layers)

    def __copy__(self):
        new_model = PerceptronModel(
            *[layer.__copy__() for layer in
              self.neural_network])
        new_model.compile()
        for new_layer, layer in zip(new_model.neural_network[1:], self.neural_network[1:]):
            for new_neuron, neuron in zip(new_layer.neurons, layer.neurons):
                new_neuron.links[0] = neuron.links[0]
                new_neuron.bias_weight = neuron.bias_weight
        return new_model

    def compile(self):
        """
        Define all links between neurons and make layer structure immutable
        """
        linked_layer = self.neural_network[0]
        for layer in self.neural_network[1:]:
            layer.linked_layer = linked_layer
            layer.set_links()
            linked_layer = layer
        self.neural_network = tuple(self.neural_network)

    def clear_values(self):
        """
        clear all neurons values, like a gradient and value
        """
        for layer in self.neural_network:
            layer.clear_data()

    def calculate(self) -> list[float]:
        """
        calculate all model layers by getting values of each neuron
        :return: values of output neurons
        """
        for layer in self.neural_network:
            layer.get_values()
        return self.neural_network[-1].get_values()

    def data_load(self, data: list, number_of_layer=0):
        """
        upload values of neurons to some layer of model (by default to 0 layer)
        :param data: uploaded values
        :param number_of_layer: number of layer wherein will be uploaded values
        """
        for neuron, data_element in zip(self.neural_network[number_of_layer].neurons, data):
            neuron.value = data_element

    def accuracy_estimate(self, estimate_set: list[list[list[float], list[float]]], loss_function=functions.true_loss):

        # TODO: Make that work in all cases
        """

                :param estimate_set: data which will be used for determination of accuracy
                :param loss_function: loss function which determines the accuracy of model
                :return: accuracy of model
                """
        accuracy = []
        for data_part in estimate_set:
            self.clear_values()
            self.data_load(data_part[0])
            accuracy_part = loss_function(self.calculate(), data_part[1])
            accuracy.append(sum(accuracy_part) / len(accuracy_part))
        return 1 - sum(map(lambda x: abs(x), accuracy)) / len(accuracy)

    def save_model(self, name_of_model: str):
        # TODO: I think i can save neuron activation function much better
        """
            save a compiled model to a JSON file
            :param name_of_model: name of model, which will be the file name
            """
        model_description = {
            "name": name_of_model,
            "num_of_layers": len(self.neural_network),
            "type_of_model": str(self.__class__.__name__)
        }
        layers = dict()
        layers = {layer_index + 1: [len(layer.neurons), str(layer.__class__.__name__),
                                    [[str(neuron.__class__.__name__), str(neuron.activation_function.__name__),
                                      neuron.bias_weight, neuron.links[0]]
                                     for
                                     neuron in layer.neurons]] for
                  layer_index, layer in
                  enumerate(self.neural_network[1:])}
        layers[0] = [len(self.neural_network[0].neurons), str(self.neural_network[0].__class__.__name__)]
        model_description["layers"] = layers
        with open(name_of_model + ".json", "w") as model_file:
            model_description = json.dumps(model_description)
            model_file.write(model_description)

    @staticmethod
    def load_model(file_source: str):
        new_model = PerceptronModel()
        with open(file_source, "r") as model_file:
            parameters = json.loads(model_file.read())
            layer_specs = parameters["layers"]["0"]
            new_model.neural_network.append(functions.str_to_class(layer_specs[1])(layer_specs[0]))
            for layer_index in range(1, parameters["num_of_layers"]):
                layer_specs = parameters["layers"][str(layer_index)]
                new_model.neural_network.append(
                    functions.str_to_class(layer_specs[1])(layer_specs[0]))
            new_model.compile()
            for layer_index in range(1, parameters["num_of_layers"]):
                layer_specs = parameters["layers"][str(layer_index)]
                for neuron_index, neuron in enumerate(new_model.neural_network[layer_index].neurons):
                    neuron.activation_function = functions.str_to_function(layer_specs[2][neuron_index][1])
                    neuron.derivative_function = functions.pairs[
                        functions.str_to_function(layer_specs[2][neuron_index][1])]
                    neuron.bias_weight = layer_specs[2][neuron_index][2]
                    neuron.links[0] = layer_specs[2][neuron_index][3]
            return new_model


class ConvolutionalModel(__Model):
    pass
