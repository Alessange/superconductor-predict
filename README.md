# Superconductor Transition Temperature Prediction via Transformer Models Applied on Crystal Lattice Data

This project primarily focuses on the application of transformer models to predict superconductor transition temperatures.

## Pre-training
Our model is pre-trained using formulation energy data. The inputs for this stage are atoms and their corresponding positions. Consider the following example:
```python
sentence = ['Ca', [0.0, 0.0, 0.0], 'Cl', [0.0, 1.2, 1.3]]
```
We have already pre-processed the dataset to convert it into string format, thereby eliminating the need for tokenization.

### Feature Embedding
- Each atom (e.g., 'Ca') is embedded into a vector of 25 dimensions. Each dimension corresponds to a specific property of the atom, such as group, mass, density, and more. **Note: All values are normalized within each dimension, regardless of unit.**
- Each atom's position is transformed using **_Gaussian distribution_**, resulting in a vector of the same shape as the atom's vector representation.
### Sentence Concatenation
We apply a **Self-Attention Mechanism** to each sentence. This allows each atom to "examine" other atoms, enabling a more nuanced understanding of the overall structure. The Self-Attention mechanism utilizes multiple heads to capture various structural aspects.
