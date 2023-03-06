import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchsummary import summary
from typing import List

# Set of possible Loss functions for the model
LOSS_FUNCTIONS = {"l1", "l2", "mse", "mae"}
# Set of possible activation functions for the model
ACTIVATION_FUNCTIONS = {"sigmoid", "tanh", "relu"}
# Set of possible optimizers for the model
OPTIMIZERS = {"adam", "sgd", "lbfgs"}

class FnDataset(Dataset):
    ''' Creates a dataset given a domain X and a sfunction on the domain
        x and y are numpy arrays 
    '''
    def __init__(self, x: np.array, y: callable):
        self.x = torch.from_numpy(x).type(torch.FloatTensor)
        self.y = torch.from_numpy(y).type(torch.FloatTensor)

    def __getitem__(self, index: int):
        ''' Retrieve a single point (x, f(x)) from the function dataset '''
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.x.shape[0]

def generate_dataset(function: callable, x: np.array) -> FnDataset:
    ''' Helper function to automatically generate an FnDataset object from a function and a domain '''
    function = np.vectorize(function)
    y = function(x)
    return FnDataset(x, y)    

class Fn(nn.Module):
    ''' A Multilayer Perceptron Regressor
        intended to match the implementation and use of sklearn's MLPRegressor Object 
    '''
    def __init__(self, sizes: List[int], activations: List[str], loss: str = "l1", optimizer: str = "adam", lr: float = 1e-5):
        if len(sizes) < len(activations):
            raise ValueError("Cannot have null layers with activation functions!")
        elif len(activations) < len(sizes):
            activations.extend([None]*(len(sizes)-len(activations)))
        if loss.lower() not in LOSS_FUNCTIONS:
            raise ValueError(f"{loss} is not a valid loss function")
        if optimizer.lower() not in OPTIMIZERS:
            raise ValueError(f"{optimizer} is not a valid optimizer")
        if any(map(lambda activation: activation not in ACTIVATION_FUNCTIONS and activation, activations)):
            raise ValueError("Invalid activation function list")

        super().__init__()
        # Dynamically construct the model
        self.model = nn.ModuleList()
        for i in range(len(sizes)-1):
            in_, out = sizes[i:i+2]
            activation = activations[i]
            if activation:
                activation = activation.lower()
            self.model.append(nn.Linear(in_, out))

            activation_function = None
            if activation == "sigmoid":
                activation_function = nn.Sigmoid()
            elif activation == "tanh":
                activation_function = nn.Tanh()
            elif activation == "relu":
                activation_function = nn.ReLU()

            if activation_function is not None:
                self.model.append(activation_function)

        self.model = nn.Sequential(*self.model)
        
        # Find device to put model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        # Optimizer
        optimizer = optimizer.lower()
        if optimizer == "adam":
            self.opt = optim.Adam(self.model.parameters(), lr=lr)
        elif optimizer == "sgd":
            self.opt = optim.SGD(self.model.parameters(), lr=lr)
        elif optimizer == "lbfgs":
            self.opt = optim.LBFGS(self.model.parameters(), lr=lr)

        # Loss Function 
        if loss == "l1" or loss == "mae":
            self.loss = nn.L1Loss()
        elif loss == "l2" or loss == "mse":
            self.loss = nn.MSELoss()

    def forward(self, x: torch.TensorType):
        ''' A single forward pass through the network '''
        if not isinstance(x, torch.Tensor):
            x = torch.Tensor([x])
        if x.device != self.device:
            x = x.to(self.device)
        return self.model(x)

    def __str__(self) -> str:
        ''' Use the torchsummary library to convert the model to a string representation '''
        return str(summary(self.model, verbose=0))

    def train(self, loader: DataLoader):
        ''' A single training iteration '''
        self.model.train()

        total_loss = 0.0
        for x, y in loader:
            self.opt.zero_grad()
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            l = self.loss(y_hat, y)
            l.backward()
            self.opt.step()
            total_loss += l.item()
    
    @torch.no_grad()
    def test(self, loader: DataLoader):
        ''' A single validation step '''
        self.model.eval()

        test_loss = 0.0
        for x, y in loader:
            x, y = x.to(self.device), y.to(self.device)
            y_hat = self.model(x)
            loss = self.loss(y_hat, y)
            test_loss += loss.item()

        print(f"Testing loss: {test_loss:.5f}")
    
    def fit(self, X: np.array, function: callable, epochs: int = 200):
        ''' Fits the model to a function given a domain (np array), and a function (callable)
        '''
        dataset = generate_dataset(function, X)
        loader = DataLoader(dataset=dataset, batch_size=1, shuffle=True)

        self.test(loader)
        for _ in range(epochs):
            self.train(loader)
            self.test(loader)

if __name__ == "__main__":
    from math import pi
    
    @np.vectorize
    def f(x):
        return np.sin(x)

    X = np.arange(0, 2*pi + 1, 0.25)

    model = Fn(sizes=[1, 1096, 1096, 1], activations=['tanh', 'tanh', 'tanh'], loss='l2')
    model.fit(X, f, epochs=500)

    y = model(0.0).item()
    print(y)

    @np.vectorize
    def model_(x):
        return model(torch.Tensor([x]).to(model.device)).detach().cpu()

    X = np.arange(-1, 3*pi / 2, 0.01)
    y = model_(X)
    plt.plot(X, y)
    plt.plot(X, f(X))

    plt.show()