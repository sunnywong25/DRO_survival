import numpy as np
from numpy.random import normal
from scipy import optimize
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

def threshplus(x):
    y = x.copy()
    y[y<0]=0
    return y

def threshplus_tensor(x):
    y = x.clone()
    y[y<0]=0
    return y
    
class LinearModel(nn.Module):
    def __init__(self, input_size, output_size, bias=True):
        super(LinearModel, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=bias)

    def forward(self, X):
        out = self.linear(X)
        return out

class DroLoss(nn.Module):
    def __init__(self, loss: nn.Module, div_family_k=2, radius_rho=0.5):
        super(DroLoss, self).__init__()
        self.eta = nn.Parameter(torch.randn(1), requires_grad=True)
        self.loss = loss
        self.k_star = div_family_k/(div_family_k - 1)
        self.c_rho = (1 + div_family_k * (div_family_k - 1) * radius_rho) ** (1/div_family_k)

    def forward(self, prediction, *targets):
        # Apply the linear layer
        loss_values = self.loss(prediction, *targets)
        out = self.c_rho * torch.mean(threshplus_tensor(loss_values - self.eta) ** self.k_star) ** (1/self.k_star) + self.eta
        # Incorporate the extra parameter into the computation
        # Here, we just add the extra parameter to the output of the linear layer (example usage)
        return out

def fit_dro(model, X_tensor, list_of_targets, dro_loss, lr=0.01, epochs=500, tol = 10e-7, patience = 5, verbose=False):
    dro_loss_trace = []
    count = 0
    optimizer = optim.Adam(list(model.parameters()) + list(dro_loss.parameters()), lr=lr)
    is_converge = False
    for epoch in range(epochs):
        # Original loss
        prediction = model(X_tensor)
        objective = dro_loss(prediction, *list_of_targets)
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        dro_loss_trace.append(objective.item())
        if verbose:
            print(f"Epoch {epoch+1} - Loss: {objective.item():.4f}")
        if epoch >= 1:
            if dro_loss_trace[-2] - dro_loss_trace[-1] < tol:
                count += 1
        if count >= patience:
            is_converge = True
            break
    print("Convergence" if is_converge else "Divergence")
    return dro_loss_trace

def fit_erm(model, X_tensor, list_of_targets, loss_indiv, lr=0.01, epochs=500, tol = 10e-7, patience = 5, verbose=False):
    loss_trace = []
    count = 0
    optimizer = optim.Adam(model.parameters(), lr=lr)
    for epoch in range(epochs):
        # Original loss
        prediction = model(X_tensor)
        objective = torch.mean(loss_indiv(prediction, *list_of_targets))
        optimizer.zero_grad()
        objective.backward()
        optimizer.step()
        loss_trace.append(objective.item())
        if verbose:
            print(f"Epoch {epoch+1} - Loss: {objective.item():.4f}")
        if epoch >= 1:
            if loss_trace[-2] - loss_trace[-1] < tol:
                count += 1
        if count >= patience:
            print("Convergence")
            break
    return loss_trace

class DroOptimizer:
    def __init__(self, loss_func, eps):
        self.loss_func = loss_func
        self.eps = eps
        self.beta = None
        self.eta = None

    def dro_loss(self, beta, eta, X, y, is_beta0_intercept=False, verbose=False, **kwargs):
        if is_beta0_intercept:
            pred = beta[0] + np.dot(X, beta[1:])
        else:
            pred = np.dot(X, beta)
        out = np.sqrt(2 * ((1.0 / self.eps - 1.0)** 2.0)+1) * np.sqrt(np.mean(threshplus(self.loss_func(pred, y, **kwargs) - eta) ** 2.0)) + eta
        if verbose:
            print(f"beta = {beta}, eta = {eta}, dro loss = {out}")
        return out

    def fit_dro(self, X, y, is_include_intercept=False, verbose=False, **kwargs):
        func = lambda para: self.dro_loss(para[:-1], para[-1], X, y, is_include_intercept, verbose, **kwargs)
        para_len = X.shape[1] + 1
        para_len += 1 if is_include_intercept else 0
        para_init = normal(size=para_len)
        self.optimizer = optimize.minimize(fun=func, x0=para_init)
        para_fitted = self.optimizer.x
        self.beta = para_fitted[:-1]
        self.eta = para_fitted[-1]

    def __call__(self, X):
        if len(self.beta) - X.shape[1] == 1:
            out = self.beta[0] + np.dot(X, self.beta[1:])
        else:
            out = np.dot(X, self.beta)
        return out


if __name__ == "__main__":
    set_random_seed()
    input_dim = 10
    output_dim = 1
    loss = nn.MSELoss(reduction="none")
    dro_loss = DroLoss(loss)
    model = LinearModel(input_dim, output_dim, bias=True)

    # Print model architecture
    print(model)

    # Example input
    input_tensor = torch.randn(5, input_dim)  # Batch of 5 samples, each of dimension 10
    output_tensor = torch.randn(5, 1)

    # Forward pass
    output = model(input_tensor)
    print("Output:", output)
    print("DRO Loss:", dro_loss(output,output_tensor))
    # Check trainable parameters
    for name, param in model.named_parameters():
        print(f"Parameter: {name}, Value: {param.data}")

    print(model.linear(input_tensor).detach())