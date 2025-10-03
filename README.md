# Mental Imagery In Vision-Language Models

## General Mathematical Foundations

This project utilizes the mathematical models of artificial neurons, multi-layer perceptrons (MLPs), attention mechanisms, and transformer architectures, as well as statistical foundations like the $t$-statistic.

### Artificial Neuron

A single artificial neuron computes a weighted sum of its inputs, followed by a non-linear activation:

$$
y = f\left(\sum_{i=1}^n w_i x_i + b \right)
$$

- $x_i$ : input features  
- $w_i$ : learned weights  
- $b$ : bias term  
- $f(\cdot)$ : activation function (e.g. ReLU, sigmoid, SiLU)

### Multi-Layer Perceptron (MLP)

An MLP stacks multiple layers of neurons, and is accountable for per-token enrichment. A hidden layer transformation is:

$$
h = f(Wx + b)
$$

The output layer is given by:

$$
y = g(Vh + c)
$$

where $W, V$ are weight matrices, $b, c$ are biases, $f$ is the hidden activation, and $g$ is the output activation.

### Scaled Dot-Product Attention

Attention allows models to focus on relevant parts of the input - meaning it powers contextualization. The core operation is:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

- $Q$: queries  
- $K$: keys  
- $V$: values  
- $d_k$: dimensionality of keys (used for scaling)

### Multi-Head Attention

To capture diverse relationships, we use multiple attention heads:

$$
\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O
$$

with

$$
\text{head}_i = \text{Attention}(QW_i^Q, \; KW_i^K, \; VW_i^V)
$$

where $W_i^Q, W_i^K, W_i^V, W^O$ are trainable projection matrices.

### t-Statistic

In statistics, the $t$-statistic is used to test hypotheses about means:

$$
t = \frac{\bar{x} - \mu}{s / \sqrt{n}}
$$

- $\bar{x}$ : sample mean  
- $\mu$ : population mean  
- $s$ : sample standard deviation  
- $n$ : sample size  

The resulting $t$ is compared against a $t$-distribution with $n-1$ degrees of freedom.

This document serves as a **mathematical reference** for the theory behind modern neural architectures.


## Setup

```bash
pip install -r requirements.txt
