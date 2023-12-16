import random


def weight_change(weight, linked_neuron, learning_rate, neuron):
    return weight - learning_rate * neuron.gradient * linked_neuron.value


def gradient_stepping(model, train_data: list[list[float], list[float]], learning_rate: float, loss_func):
    sample_train_data = train_data
    model.clear_values()
    model.data_load(sample_train_data[0])
    model.calculate()
    local_errors = loss_func(model.neural_network[-1].get_values(), sample_train_data[1])
    for neuron in model.neural_network[-1].neurons:  # train output_neurons
        local_error = local_errors[model.neural_network[-1].neurons.index(neuron)]  # optimize that
        neuron.update_gradient(local_error)
        neuron.links[0] = [weight_change(weight, linked_neuron, learning_rate, neuron) for weight, linked_neuron in
                           zip(neuron.links[0], neuron.links[1])]
        neuron.bias_weight = neuron.bias_weight - learning_rate * neuron.gradient
    for layer in model.neural_network[-2:-len(model.neural_network):-1]:
        for neuron in layer.neurons:
            local_error = sum(
                [linked_neuron.links[0][linked_neuron.links[1].index(
                    neuron)] * linked_neuron.gradient if linked_neuron.gradient is not None else 0 for
                 linked_neuron in
                 neuron.linked_neurons])
            neuron.update_gradient(local_error)
            neuron.links[0] = [weight_change(weight, linked_neuron, learning_rate, neuron) for weight, linked_neuron in
                               zip(neuron.links[0], neuron.links[1])]
            neuron.bias_weight = neuron.bias_weight - learning_rate * neuron.gradient


def sgd_optimize(model, train_data: list[list[list[float], list[float]]], learning_rate: float, epochs: int,
                 loss_func):
    for epoch in range(epochs):
        gradient_stepping(model, random.choice(train_data), learning_rate, loss_func)


def minibatch_optimize(model, train_data: list[list[float], list[float]], learning_rate: float,
                       batch_size: int,
                       epochs: int,
                       loss_func):
    for epoch in range(epochs):
        batch = random.choices(train_data, k=batch_size)
        gradient_map = [[[] for gradient in layer.neurons] for layer in model.neural_network]
        values_map = [[[] for value in layer.neurons] for layer in model.neural_network]
        for data in batch:
            new_model = model.__copy__()
            gradient_stepping(new_model, data, learning_rate, loss_func)
            local_gradients = [new_layer.get_gradients() for new_layer in new_model.neural_network]
            local_values = [new_layer.get_values() for new_layer in new_model.neural_network]
            for layer_index, layer_gradients in enumerate(local_gradients):
                for neuron_index, neuron_gradient in enumerate(layer_gradients):
                    gradient_map[layer_index][neuron_index].append(neuron_gradient)
            for layer_index, layer_values in enumerate(local_values):
                for neuron_index, neuron_value in enumerate(layer_values):
                    values_map[layer_index][neuron_index].append(neuron_value)
        for neuron_index, neuron in enumerate(model.neural_network[0].neurons):
            value_list = values_map[0][neuron_index]
            neuron.value = sum(value_list) / len(value_list)
        for layer_index, layer in enumerate(model.neural_network[1:]):
            for neuron_index, neuron in enumerate(layer.neurons):
                value_list = values_map[layer_index + 1][neuron_index]
                gradient_list = gradient_map[layer_index + 1][neuron_index]
                neuron.gradient = sum(gradient_list) / len(gradient_list)
                neuron.value = sum(value_list) / len(value_list)
                neuron.links[0] = [weight_change(weight, linked_neuron, learning_rate, neuron) for weight, linked_neuron
                                   in
                                   zip(neuron.links[0], neuron.links[1])]
                neuron.bias_weight = neuron.bias_weight - learning_rate * neuron.gradient


def batch_optimize(model, train_data: list[list[list[float], list[float]]], learning_rate: float, epochs: int,
                   loss_func):
    for epoch in range(epochs):
        batch = random.choices(train_data, k=len(train_data))
        gradient_map = [[[] for gradient in layer.neurons] for layer in model.neural_network]
        values_map = [[[] for value in layer.neurons] for layer in model.neural_network]
        for data in batch:
            new_model = model.__copy__()
            gradient_stepping(new_model, data, learning_rate, loss_func)
            local_gradients = [new_layer.get_gradients() for new_layer in new_model.neural_network[1:]]
            local_values = [new_layer.get_values() for new_layer in new_model.neural_network]
            for layer_index, layer_gradients in enumerate(local_gradients):
                for neuron_index, neuron_gradient in enumerate(layer_gradients):
                    gradient_map[layer_index + 1][neuron_index].append(neuron_gradient)
            for layer_index, layer_values in enumerate(local_values):
                for neuron_index, neuron_value in enumerate(layer_values):
                    values_map[layer_index][neuron_index].append(neuron_value)
        for neuron_index, neuron in enumerate(model.neural_network[0].neurons):
            value_list = values_map[0][neuron_index]
            neuron.value = sum(value_list) / len(value_list)
        for layer_index, layer in enumerate(model.neural_network[1:]):
            for neuron_index, neuron in enumerate(layer.neurons):
                value_list = values_map[layer_index + 1][neuron_index]
                gradient_list = gradient_map[layer_index + 1][neuron_index]
                neuron.gradient = sum(gradient_list) / len(gradient_list)
                neuron.value = sum(value_list) / len(value_list)
                neuron.links[0] = [weight_change(weight, linked_neuron, learning_rate, neuron) for weight, linked_neuron
                                   in
                                   zip(neuron.links[0], neuron.links[1])]
                neuron.bias_weight = neuron.bias_weight - learning_rate * neuron.gradient


def momentum():
    pass


optimizers = {
    "sgd": sgd_optimize,
    "minibatch": minibatch_optimize,
    "batch": batch_optimize
}
