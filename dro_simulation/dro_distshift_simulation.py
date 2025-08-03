import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable 

from dro_models import *
from numpy.random import normal, binomial

def simulate_distshift_dgp(beta, sim_n=200):
    sim_X = normal(size=(sim_n, len(beta)))
    sim_y = 2.0 * (np.dot(sim_X, beta) > 0) - 1.0
    flip = binomial(1, 0.1, sim_n)
    sim_y[flip] *= -1.0
    sim_y = sim_y.reshape(-1,1)
    return sim_X, sim_y

class HingeLoss(nn.Module):
    def __init__(self):
        super(HingeLoss, self).__init__()

    def forward(self, prediction, target):
        out = threshplus_tensor(1 - target * prediction)
        return out

if __name__ == "__main__":
    eval_test = lambda y, pred: threshplus(1 - y * pred)
    
    # Simulation config
    random_state = 10
    n_cov = 5
    sim_n_train = 100
    sim_n_test = 100000

    # Grids
    shift_radian_candidates = np.linspace(0, 3.0, 20)
    k_candidates = [1.5, 2.0, 4.0]
    rho = 0.5

    # Model training parameters
    bias = False
    epochs = 3000 # max epochs
    lr = 0.01
    verbose = False

    # Initializations
    set_random_seed(random_state)
    beta = normal(size = n_cov)
    beta /= np.sqrt(np.dot(beta, beta))

    v = normal(size = n_cov)
    v[0] -= np.dot(beta, v)/beta[0]
    v /= np.sqrt(np.dot(v, v))
    avg_test_loss = np.zeros((len(k_candidates) + 1, len(shift_radian_candidates)))
    quantile90 = np.zeros((len(k_candidates) + 1, len(shift_radian_candidates)))
    beta_fitted = np.zeros((len(k_candidates) + 1, len(beta)))
    eta_fitted = np.zeros(len(k_candidates))

    # Simulation
    # Generate datasets
    sim_X_train, sim_y_train = simulate_distshift_dgp(beta, sim_n=sim_n_train)

    # Convert np.array to torch.tensor if using torch implementation
    sim_X_train_tensor = Variable((torch.from_numpy(sim_X_train)).float())
    sim_y_train_tensor = Variable((torch.from_numpy(sim_y_train)).float())

    # ERM baseline
    model = LinearModel(len(beta), 1, bias)
    loss_indiv = HingeLoss()
    fit_erm(model=model, 
            X_tensor=sim_X_train_tensor, 
            list_of_targets=[sim_y_train_tensor], 
            loss_indiv=loss_indiv, 
            epochs=epochs,
            lr=lr,
            verbose=verbose)
    beta_fitted[-1, :] = [para.detach().numpy() for para in model.parameters()][0]

    # test_loss_indiv = eval_test(sim_y_test.reshape(-1,), pred_test)
    # erm_avg_test_loss = test_loss_indiv.mean()
    # erm_loss_on_minor = test_loss_indiv[G_test==0].mean()

    for k_ind, k in enumerate(k_candidates):
        print(f"{k:.2f}: ", end = '')

        # torch implementation
        model = LinearModel(len(beta), 1, bias)
        loss_indiv = HingeLoss()
        dro_loss = DroLoss(loss_indiv, div_family_k=k, radius_rho=rho)
        dro_loss_trace = fit_dro(model=model, 
                                X_tensor=sim_X_train_tensor, 
                                list_of_targets=[sim_y_train_tensor], 
                                dro_loss=dro_loss, 
                                epochs=epochs,
                                lr=lr,
                                verbose=verbose)
        beta_fitted[k_ind, :] = [para.detach().numpy() for para in model.parameters()][0]
        eta_fitted[k_ind] = [para.detach().numpy() for para in dro_loss.parameters()][0]

    for shift_ind, shift_radian in enumerate(shift_radian_candidates):
        beta_shifted = beta*np.cos(shift_radian) + v*np.sin(shift_radian)
        sim_X_test, sim_y_test = simulate_distshift_dgp(beta_shifted, sim_n=sim_n_test)
        for k_ind in range(len(k_candidates) + 1):
            pred_test = np.dot(sim_X_test, beta_fitted[k_ind, :])
            test_loss_indiv = eval_test(sim_y_test.reshape(-1,), pred_test)
            avg_test_loss[k_ind, shift_ind] = test_loss_indiv.mean()
            quantile90[k_ind, shift_ind] = np.quantile(test_loss_indiv, 0.9)

    # Plots
    situation = 'distshift'
    marker_list = ['v', 's', '*']
    color_list = ['green', 'blue', 'black']
    # for i in range(len(beta)):
    #     plt.figure()
    #     plt.plot(plot_x, [beta[i]]*len(rho_candidates), label = r'True Major $\beta_{}$'.format(i+1), linestyle='--', color = 'orange')
    #     plt.plot(plot_x, [beta_minor[i]]*len(rho_candidates), label = r'True Minor $\beta_{}$'.format(i+1), linestyle='--', color = 'purple')
    #     for k_ind, k in enumerate(k_candidates):
    #         plt.plot(plot_x, beta_fitted[k_ind,:,i], label = f'k = {k:.1f}', marker=marker_list[k_ind], color=color_list[k_ind])
    #     plt.xlabel(r'log$_{10}\rho$')
    #     plt.ylabel(r'$\beta_{}$'.format(i+1))
    #     plt.legend()
    #     plt.title(f"Beta {i+1}")
    #     plt.savefig(f'plots/{situation}_beta{i+1}.png')
    #     plt.close()

    plt.figure()
    plt.plot(shift_radian_candidates, avg_test_loss[-1,:], label = 'ERM', color = 'red')
    plt.plot(shift_radian_candidates, quantile90[-1,:], color = 'red', linestyle='--')
    for k_ind, k in enumerate(k_candidates):
        plt.plot(shift_radian_candidates, avg_test_loss[k_ind,:], label = f'k = {k:.1f}', color = color_list[k_ind])
        plt.plot(shift_radian_candidates, quantile90[k_ind,:], color = color_list[k_ind], linestyle='--')
    plt.xlabel('angle')
    plt.ylabel('loss')
    plt.legend()
    plt.title("Average Loss and 90% Quantile Loss")
    plt.savefig(f'plots/{situation}_loss.png')
    plt.close()

    # plt.figure()
    # plt.plot(plot_x, [erm_loss_on_minor]*len(rho_candidates), label = 'ERM', marker='o', color = 'red')
    # for k_ind, k in enumerate(k_candidates):
    #     plt.plot(plot_x, loss_on_minor[k_ind,:], label = f'k = {k:.1f}', marker=marker_list[k_ind], color = color_list[k_ind])
    # plt.xlabel(r'log$_{10}\rho$')
    # plt.ylabel('loss on minority')
    # plt.ylim(bottom=0.0)
    # plt.legend()
    # plt.title('Loss on Minority')
    # plt.savefig(f'plots/{situation}_loss_on_minority.png')
    # plt.close()