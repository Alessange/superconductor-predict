# Predicting Superconductor Transition Temperature via Transformer
The project is mainly about using transformer for predicting superconductor transition temperature.


## Pre-training
In this work, we pretrain the model on formulation energy data with atoms and corresponding positions as input.
e.g.,
```
sentence = [['He', [0.0, 0.0, 0.0], 'H', [0.0, 1.2, 1.3]]
```
