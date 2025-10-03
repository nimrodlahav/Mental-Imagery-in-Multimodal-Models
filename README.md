# Mental Imagery In Vision-Language Models

## Mathematical Foundations

This project explores the mathematical underpinnings of artificial neurons, multi-layer perceptrons (MLPs), attention mechanisms, and transformer architectures, as well as statistical foundations like the $t$-statistic.

## Artificial Neuron

A single artificial neuron computes a weighted sum of its inputs, followed by a non-linear activation:

$$
y = f\!\left(\sum_{i=1}^n w_i x_i + b \right)
$$

- $x_i$ : input features  
- $w_i$ : learned weights  
- $b$ : bias term  
- $f(\cdot)$ : activation function (e.g. ReLU, sigmoid, tanh)

## Multi-Layer Perceptron (MLP)

An MLP stacks multiple layers of neurons. A hidden layer transformation is:

$$
h = f(Wx + b)
$$

The output layer is given by:

$$
y = g(Vh + c)
$$

where $W, V$ are weight matrices, $b, c$ are biases, $f$ is the hidden activation, and $g$ is the output activation.

## Scaled Dot-Product Attention

Attention allows models to focus on relevant
