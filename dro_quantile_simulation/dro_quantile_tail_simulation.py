import numpy as np

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable 

from dro_models import *
from numpy.random import normal

def simulate_tail_dgp(beta, beta1_shift=1.0, sigma=0.1, sim_n=200):
    sim_X = normal(size=(sim_n, len(beta)))
    sim_X1 = sim_X[:,0]
    sim_y = np.dot(sim_X, beta) + (sim_X1 > 1.645)*beta1_shift*sim_X1 + np.random.normal(loc=0, scale=sigma, size=sim_n)
    sim_y = sim_y.reshape(-1,1)
    return sim_X, sim_y
    
if __name__ == "__main__":
    eval_test = lambda y, pred: 0.5*(y - pred)**2
    
    # Simulation config
    random_state = 10
    sim_n_train = 2000
    n_cov = 5
    sim_n_test = 100000
    beta1_shift = 1.0
    sigma = 0.1
    minor_prop = 0.1

    # Grids
    k_candidates = [1.5, 2.0, 4.0]
    rho_candidates = np.array([0.001, 0.01, 0.1, 0.5, 4.5])

    # Model training parameters
    bias = False
    epochs = 10000 # max epochs
    lr = 0.01
    verbose = False

    # Initializations
    set_random_seed(random_state)
    beta = normal(size = n_cov)
    beta /= np.sqrt(np.dot(beta, beta))
    print(f"True beta: {beta}")
    major_loss = np.zeros((len(k_candidates) + 1, len(rho_candidates)))
    minor_loss = np.zeros((len(k_candidates) + 1, len(rho_candidates)))
    beta_fitted = np.zeros((len(k_candidates) + 1, len(rho_candidates), n_cov))
    eta_fitted = np.zeros((len(k_candidates), len(rho_candidates)))

    # Simulation
    # Generate datasets
    sim_X_train, sim_y_train = simulate_tail_dgp(beta, beta1_shift, sigma, sim_n_train)
    sim_X_test, sim_y_test = simulate_tail_dgp(beta, beta1_shift, sigma, sim_n_test)
    is_minor_test = (sim_X_test[:,0] > 1.645)

    # Convert np.array to torch.tensor if using torch implementation
    sim_X_train_tensor = Variable((torch.from_numpy(sim_X_train)).float())
    sim_y_train_tensor = Variable((torch.from_numpy(sim_y_train)).float())
    sim_X_test = torch.from_numpy(sim_X_test).float()

    # ERM baseline
    model = LinearModel(n_cov, 1, bias)
    loss_indiv = nn.MSELoss(reduction='none')
    fit_erm(model=model, 
            X_tensor=sim_X_train_tensor, 
            list_of_targets=[sim_y_train_tensor], 
            loss_indiv=loss_indiv, 
            epochs=epochs,
            lr=lr,
            verbose=verbose)
    beta_fitted[-1, :, :] = [para.detach().numpy() for para in model.parameters()][0]
    pred_test = model(sim_X_test).detach().numpy().reshape((-1,))
    test_loss_indiv = eval_test(sim_y_test.reshape(-1,), pred_test)
    major_loss[-1,:] = test_loss_indiv[~is_minor_test].mean()
    minor_loss[-1,:] = test_loss_indiv[is_minor_test].mean()

    for k_ind, k in enumerate(k_candidates):
        for rho_ind, rho in enumerate(rho_candidates):
            print(f"{rho:.2f}: ", end = '')

            # torch implementation
            model = LinearModel(n_cov, 1, bias)
            loss_indiv = nn.MSELoss(reduction="none")
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
            major_loss[k_ind, rho_ind] = test_loss_indiv[~is_minor_test].mean()
            minor_loss[k_ind, rho_ind] = test_loss_indiv[is_minor_test].mean()
            # test_loss_indiv = eval_test(sim_y_test.reshape(-1,), pred_test)
            # avg_test_loss[k_ind, rho_ind] = test_loss_indiv.mean()
            # loss_on_minor[k_ind, rho_ind] = test_loss_indiv[G_test==0].mean()
    print(eta_fitted)
    # Plots
    situation = 'tail'
    marker_list = ['v', 's', '*']
    color_list = ['green', 'blue', 'black']
    plot_x = np.log(rho_candidates)/np.log(10.0)
    for i in range(len(beta)):
        plt.figure()
        for k_ind, k in enumerate(k_candidates):
            plt.plot(plot_x, beta_fitted[k_ind,:,i], label = f'k = {k:.1f}', marker=marker_list[k_ind], color=color_list[k_ind])
        plt.xlabel(r'log$_{10}\rho$')
        plt.ylabel(r'$\beta_{}$'.format(i+1))
        plt.legend()
        plt.title(f"Beta {i+1}")
        plt.savefig(f'plots/{situation}_beta{i+1}.png')
        plt.show()
        plt.close()

    plt.figure()
    plt.plot(plot_x, major_loss[-1,:], label='ERM', marker='o', color = 'red')
    plt.plot(plot_x, minor_loss[-1,:], marker='o', color = 'red', linestyle = '--')
    for k_ind, k in enumerate(k_candidates):
        plt.plot(plot_x, major_loss[k_ind,:], label = f'k = {k:.1f}', marker=marker_list[k_ind], color = color_list[k_ind])
        plt.plot(plot_x, minor_loss[k_ind,:], marker=marker_list[k_ind], color = color_list[k_ind], linestyle = '--')
    plt.xlabel(r'log$_{10}\rho$')
    plt.ylabel('average loss')
    plt.legend()
    plt.title("Average Loss")
    plt.savefig(f'plots/{situation}_major_vs_minor.png')
    plt.show()
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