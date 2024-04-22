Fergana is machine learning framework


## Install


```
$ pip install fergana
```


#### *Your first Fergana model*


```Python

from fergana.structures import models, layers

Fergana = models.PerceptronModel(
    layers.InputLayer(4),
    layers.HiddenLayer(4),
    layers.OutputLayer(3)
)
Fergana.compile()
Fergana.data_load([5.1, 3.5, 1.4, 0.2])
print(Fergana.calculate())

```
