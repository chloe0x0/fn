# fn
Function approximation with Neural Networks! 

Using shallow neural networks to learn mathematical functions.
Trigonometric functions, rational functions, polynomials, etc etc
Even boolean functions! (build a small CPU with neural networks as the logical gates?)

This is currently intended to function like sklearn's MLPRegressor object.


## Example, learning sin(x)
```python
import numpy as np
from fn import Fn

f = np.sin
X = np.arange(-5, 5, 0.25)
model = Fn(sizes=[1, 1096, 1096, 1096, 1], activations=['tanh', 'tanh', 'tanh'], loss='l1', optimizer="adam")
model.fit(X, f, epochs=750)

@np.vectorize
def model_(x):
    return model(torch.Tensor([x]).to(model.device)).detach().cpu()
    
y = model_(X)
plt.plot(X, y)
plt.plot(X, f(X))

plt.show()
```
the model will look like

![](imgs/Figure_1.png)

## Goal
Construct a framework where a user can input a function, and a neural network is automatically created to learn the function.
The goal is ultimatley to use as few neurons and hidden layers as possible

# Theory
The Universal Approximation Theorem states that a neural network with a single hidden layer to approximate any continious function on a closed interval to arbitrary error
