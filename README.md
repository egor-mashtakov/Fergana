Fergana is a machine learning framework that allows you to do some fancy things with numbers and work with deep learning in several ways.



# Install


*Installation via pip*
```
$ pip install fergana
```
*Installation via poetry*
```
$ poetry add fergana
```
### *Your first Fergana programm*

```shell
$ python
```

```python
>>> from fergana.math import functions as mf
>>> result = mf.linear(1)
>>> print(result)
1
```

### *Your first Fergana model*
```Python
from fergana.structures import models, layers

Fergana = models.PerceptronModel(
    layers.InputLayer(2),
    layers.HiddenLayer(3),
    layers.OutputLayer(2)
)
Fergana.compile()
Fergana.data_load([1.0, 0.0])
print(Fergana.calculate())
```

*the output should look something like this:*

```
[0.7349377405283132, 0.3685998574341806]
```
This model doesn't actually do anything, it's just random numbers that we get by processing the input data through a neural network. Let's train it to swap input values ​​and output them, for example: 

[1.0, 0.0] -> [0.0, 1.0]

To do this, we will use one of the simplest optimizers SGD (Stochastic Gradient Descent), this is what the code will look like:

```Python
from fergana.structures import models, layers
from fergana.math import optimizers, functions

Fergana = models.PerceptronModel(
    layers.InputLayer(2),
    layers.HiddenLayer(3),
    layers.OutputLayer(2)
)
Fergana.compile()
Fergana.data_load([1.0, 0.0])
print(Fergana.calculate())
train_dataset = [[[1.0, 0.0], [0.0, 1.0]],
                 [[0.0, 1.0], [1.0, 0.0]],
                 [[0.4, 0.7], [0.7, 0.4]],
                 [[0.1, 0.5], [0.5, 0.1]]]

optimizers.sgd_optimize(Fergana, train_dataset, 1, 1000, loss_func=functions.true_loss)
Fergana.data_load([1.0, 0.0])
print(Fergana.calculate())

```

*and the final output should look like  this:*
```
[0.5518152032198591, 0.623856090521113]
[0.04820617410256474, 0.9601331009350044]
```

As we can see, the end result is pretty close to our task, so congratulations: you've just trained your first neural network!
