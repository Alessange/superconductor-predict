# Superconductor Transition Temperature Prediction via Transformer Models Applied on Crystal Lattice Data

This project primarily focuses on the application of transformer models to predict superconductor transition temperatures.
## Model 1: Sentence-Structure Transformer(SST)
This model is considered input be sentence-structure of crystal, for example
```python
sentence = ['Ca', [0.0, 0.0, 0.0], 'Cl', [0.0, 1.2, 1.3]]
```
### Pre-training
Our model is pre-trained using formulation energy data. The inputs for this stage are atoms and their corresponding positions. Consider the following example:
```python
sentence = ['Ca', [0.0, 0.0, 0.0], 'Cl', [0.0, 1.2, 1.3]]
```
We have already pre-processed the dataset to convert it into string format, thereby eliminating the need for tokenization.

#### Feature Embedding
- Each atom (e.g., 'Ca') is embedded into a vector of 25 dimensions. Each dimension corresponds to a specific property of the atom, such as group, mass, density, and more. **Note: All values are normalized within each dimension, regardless of unit.**
- Each atom's position is transformed using **_Gaussian distribution_**, resulting in a vector of the same shape as the atom's vector representation.
#### Sentence Concatenation
We apply a **Self-Attention Mechanism** to each sentence. This allows each atom to "examine" other atoms, enabling a more nuanced understanding of the overall structure. The Self-Attention mechanism utilizes multiple heads to capture various structural aspects.

### Model Architecture
The following drawing is the main components and layers of the model consisting of the embedding layer, transformer encoder layer, and regression head for the regression task.
![image](https://github.com/Alessange/superconductor-predict/assets/56106326/6b3b459e-976b-4795-930e-832177df84c9)

## Model 2 CGCNN embedding Transformer(CGET)
This model is built with the cgcnn as embedding for input crystal structure, and the dataset consists of the ```id.cif``` structure for representing the input crystal structure.
**Note: the pre-train dataset is the same as the SST**

### Model Architecture
The following drawing is the main components and layers of the model consisting of the embedding layer, transformer encoder layer, and regression head for the regression task.
![model2](https://github.com/Alessange/superconductor-predict/assets/56106326/58d349ae-fa00-4134-81f2-d8d0ee7e5e2a)

