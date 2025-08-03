import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

from scipy.stats import norm
from torch.autograd import Variable 
from dro_models import *
from numpy.random import normal, binomial

#The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

def simulate_quantile_subpop_dgp(beta_major, beta_minor, tau, sigma = 0.1, sim_n=200, G_dist = lambda n: binomial(1, 0.9, n)):
    sim_X = normal(size=(sim_n, len(beta_major)))
    G = G_dist(sim_n)
    pred = G*np.dot(sim_X, beta_major) + (1 - G)*np.dot(sim_X, beta_minor)
    sim_y = pred + normal(scale=sigma, size=sim_n) - norm.ppf(tau, scale=sigma)
    sim_y = sim_y.reshape((-1,1))
    return sim_X, sim_y, G

class QuantileLoss(nn.Module):
    def __init__(self, tau):
        super(QuantileLoss, self).__init__()
        self.tau = tau

    def forward(self, prediction, target):
        error = target - prediction
        return torch.max((self.tau - 1) * error, self.tau * error)

if __name__ == "__main__":
    tau = 0.5
    eval_test = lambda y, pred: (y - pred)*(tau - (y <= pred))
    
    # Simulation config
    random_state = 10
    sim_n_train = 1000
    sim_n_test = 100000
    beta_major = np.array([1,1])
    beta_minor = np.array([1,0.1])
    sigma = 0.1
    minor_prop = 0.1
    G_dist = lambda n: binomial(1, 1 - minor_prop, n)
    #G_dist = lambda n: np.random.beta(1.0, 2.0/3.0, n)

    # Grids
    k_candidates = [1.5, 2.0, 4.0]
    rho_candidates = 10**np.array([-2.0, -1.7, -1.0, -0.3, 0.0, 0.6])

    # Model training parameters
    bias = False
    epochs = 3000 # max epochs
    lr = 0.01
    verbose = False

    # Initializations
    set_random_seed(random_state)
    avg_test_loss = np.zeros((len(k_candidates), len(rho_candidates)))
    loss_on_minor = np.zeros((len(k_candidates), len(rho_candidates)))
    beta_fitted = np.zeros((len(k_candidates), len(rho_candidates), len(beta_major)))
    eta_fitted = np.zeros((len(k_candidates), len(rho_candidates)))

    # Simulation
    # Generate datasets
    sim_X_train, sim_y_train, G_train = simulate_quantile_subpop_dgp(beta_major, beta_minor, tau, sigma=sigma, sim_n=sim_n_train, G_dist=G_dist)
    sim_X_test, sim_y_test, G_test = simulate_quantile_subpop_dgp(beta_major, beta_minor, tau, sigma=sigma, sim_n=sim_n_test, G_dist=G_dist)

    # Convert np.array to torch.tensor if using torch implementation
    sim_X_train_tensor = Variable((torch.from_numpy(sim_X_train)).float())
    sim_y_train_tensor = Variable((torch.from_numpy(sim_y_train)).float())
    sim_X_test = torch.from_numpy(sim_X_test).float()

    # ERM baseline
    model = LinearModel(len(beta_major), 1, bias)
    loss_indiv = QuantileLoss(tau)
    fit_erm(model=model, 
            X_tensor=sim_X_train_tensor, 
            list_of_targets=[sim_y_train_tensor], 
            loss_indiv=loss_indiv, 
            epochs=epochs,
            lr=lr,
            verbose=verbose)
    pred_test = model(sim_X_test).detach().numpy().reshape((-1,))
    test_loss_indiv = eval_test(sim_y_test.reshape(-1,), pred_test)
    erm_avg_test_loss = test_loss_indiv.mean()
    erm_loss_on_minor = test_loss_indiv[G_test==0].mean()

    for k_ind, k in enumerate(k_candidates):
        for rho_ind, rho in enumerate(rho_candidates):
            print(f"{rho:.2f}:")

            # torch implementation
            model = LinearModel(len(beta_major), 1, bias)
            loss_indiv = QuantileLoss(tau)
            dro_loss = DroLoss(loss_indiv, div_family_k=k, radius_rho=rho)
            dro_loss_trace = fit_dro(model=model, 
                                    X_tensor=sim_X_train_tensor, 
                                    list_of_targets=[sim_y_train_tensor], 
                                    dro_loss=dro_loss, 
                                    epochs=epochs,
                                    lr=lr,
                                    verbose=verbose)
            beta_fitted[k_ind, rho_ind, :] = [para.detach().numpy() for para in model.parameters()][0]
            eta_fitted[k_ind, rho_ind] = [para.detach().numpy() for para in dro_loss.parameters()][0]
            pred_test = model(sim_X_test).detach().numpy().reshape((-1,))
            test_loss_indiv = eval_test(sim_y_test.reshape(-1,), pred_test)
            avg_test_loss[k_ind, rho_ind] = test_loss_indiv.mean()
            loss_on_minor[k_ind, rho_ind] = test_loss_indiv[G_test==0].mean()

    # Plots
    # plt.plot(plot_x, beta_fitted[:,0], label = 'beta1', color='red')
    # plt.plot(plot_x, beta_fitted[:,1], label = 'beta2', color='green')
    # plt.axhline(y=beta_minor[0], color='red', linestyle='--', label='minority beta1')
    # plt.axhline(y=beta_minor[1], color='green', linestyle='--', label='minority beta2')
    # plt.xlabel('rho')
    # plt.ylabel('beta')
    # plt.legend()
    # plt.show()
    
    situation = 'two_subpop'
    marker_list = ['v', 's', '*']
    color_list = ['green', 'blue', 'black']
    plot_x = np.log(rho_candidates)/np.log(10.0)
    for i in range(len(beta_major)):
        plt.figure()
        plt.plot(plot_x, [beta_major[i]]*len(rho_candidates), label = r'True Major $\beta_{}$'.format(i+1), linestyle='--', color = 'orange')
        plt.plot(plot_x, [beta_minor[i]]*len(rho_candidates), label = r'True Minor $\beta_{}$'.format(i+1), linestyle='--', color = 'purple')
        for k_ind, k in enumerate(k_candidates):
            plt.plot(plot_x, beta_fitted[k_ind,:,i], label = f'k = {k:.1f}', marker=marker_list[k_ind], color=color_list[k_ind])
        plt.xlabel(r'log$_{10}\rho$')
        plt.ylabel(r'$\beta_{}$'.format(i+1))
        plt.legend()
        plt.title(f"Beta {i+1}")
        plt.savefig(f'plots/{situation}_beta{i+1}.png')
        plt.close()
    
    plt.figure()
    plt.plot(plot_x, [erm_avg_test_loss]*len(rho_candidates), label = 'ERM', marker='o', color = 'red')
    for k_ind, k in enumerate(k_candidates):
        plt.plot(plot_x, avg_test_loss[k_ind,:], label = f'k = {k:.1f}', marker=marker_list[k_ind], color = color_list[k_ind])
    plt.xlabel(r'log$_{10}\rho$')
    plt.ylabel('average loss')
    plt.ylim((0.0, 0.35))
    plt.legend()
    plt.title("Average Loss")
    plt.savefig(f'plots/{situation}_average_loss.png')
    plt.close()

    plt.figure()
    plt.plot(plot_x, [erm_loss_on_minor]*len(rho_candidates), label = 'ERM', marker='o', color = 'red')
    for k_ind, k in enumerate(k_candidates):
        plt.plot(plot_x, loss_on_minor[k_ind,:], label = f'k = {k:.1f}', marker=marker_list[k_ind], color = color_list[k_ind])
    plt.xlabel(r'log$_{10}\rho$')
    plt.ylabel('loss on minority')
    plt.ylim(bottom=0.0)
    plt.legend()
    plt.title('Loss on Minority')
    plt.savefig(f'plots/{situation}_loss_on_minority.png')
    plt.close()