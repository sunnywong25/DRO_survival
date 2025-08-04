import numpy as np
import pickle
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
    
def simulate_tail_train_test(beta, beta1_shift=1.0, sigma=0.1, sim_n_train=2000, sim_n_test=100000):
    sim_X_train, sim_y_train = simulate_tail_dgp(beta, beta1_shift, sigma, sim_n_train)
    sim_X_test, sim_y_test = simulate_tail_dgp(beta, beta1_shift, sigma, sim_n_test)
    is_minor_test = (sim_X_test[:,0] > 1.645)

    # Convert np.array to torch.tensor if using torch implementation
    sim_X_train_tensor = Variable((torch.from_numpy(sim_X_train)).float())
    sim_y_train_tensor = Variable((torch.from_numpy(sim_y_train)).float())
    sim_X_test = torch.from_numpy(sim_X_test).float()

    return sim_X_train_tensor, sim_y_train_tensor, sim_X_test, sim_y_test, is_minor_test

def dump_pickle(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

if __name__ == "__main__":
    eval_test = lambda y, pred: 0.5*(y - pred)**2
    is_show_plot = True
    is_save_plot = False
    dump_pickle_path = 'C:\\Users\\Administrator\\Desktop\\DRO_survival\\dro_simulation\\data'
    situation = 'tail'
    marker_list = ['v', 's', '*']
    color_list = ['green', 'blue', 'black']

    # Simulation config
    random_state = 10
    sim_n_runs = 30
    sim_n_train = 2000
    n_cov = 5
    sim_n_test = 100000
    beta1_shift = 1.0
    sigma = 0.1

    # Grids
    k_candidates = [1.5, 2.0, 4.0]
    rho_candidates = np.array([0.001, 0.01, 0.1, 0.5, 4.5])
    plot_x = np.log(rho_candidates)/np.log(10.0)

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
    major_loss = np.zeros((sim_n_runs, len(k_candidates) + 1, len(rho_candidates)))
    minor_loss = np.zeros((sim_n_runs, len(k_candidates) + 1, len(rho_candidates)))
    beta_fitted = np.zeros((sim_n_runs, len(k_candidates) + 1, len(rho_candidates), n_cov))
    eta_fitted = np.zeros((sim_n_runs, len(k_candidates), len(rho_candidates)))

    for run_ind, run in enumerate(range(sim_n_runs)):
        print(f"Run {run_ind + 1}/{sim_n_runs}")
        # Simulation
        # Generate datasets
        sim_X_train_tensor, sim_y_train_tensor, sim_X_test, sim_y_test, is_minor_test = simulate_tail_train_test(beta, beta1_shift, sigma, sim_n_train, sim_n_test)

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
        beta_fitted[run_ind,-1,:,:] = [para.detach().numpy() for para in model.parameters()][0]
        pred_test = model(sim_X_test).detach().numpy().reshape((-1,))
        test_loss_indiv = eval_test(sim_y_test.reshape(-1,), pred_test)
        major_loss[run_ind,-1,:] = test_loss_indiv[~is_minor_test].mean()
        minor_loss[run_ind,-1,:] = test_loss_indiv[is_minor_test].mean()

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
                beta_fitted[run_ind, k_ind, rho_ind, :] = [para.detach().numpy() for para in model.parameters()][0]
                eta_fitted[run_ind, k_ind, rho_ind] = [para.detach().numpy() for para in dro_loss.parameters()][0]
                pred_test = model(sim_X_test).detach().numpy().reshape((-1,))
                test_loss_indiv = eval_test(sim_y_test.reshape(-1,), pred_test)
                major_loss[run_ind, k_ind, rho_ind] = test_loss_indiv[~is_minor_test].mean()
                minor_loss[run_ind, k_ind, rho_ind] = test_loss_indiv[is_minor_test].mean()
                # test_loss_indiv = eval_test(sim_y_test.reshape(-1,), pred_test)
                # avg_test_loss[k_ind, rho_ind] = test_loss_indiv.mean()
                # loss_on_minor[k_ind, rho_ind] = test_loss_indiv[G_test==0].mean()
    for name in ['major_loss', 'minor_loss', 'beta_fitted', 'eta_fitted']:
        dump_pickle(eval(name), f'{dump_pickle_path}/{situation}_{name}_.pkl')
        print(f"Dumped {name} to data/situation_{name}_.pkl")
    beta_fitted_mean = beta_fitted.mean(axis=0)
    beta_fitted_std = beta_fitted.std(axis=0)
    major_loss_mean = major_loss.mean(axis=0)
    major_loss_std = major_loss.std(axis=0)
    minor_loss_mean = minor_loss.mean(axis=0)
    minor_loss_std = minor_loss.std(axis=0)

    # Plots
    # for i in range(len(beta)):
    #     plt.figure()
    #     if i == 0:
    #         plt.plot(plot_x, [beta1_shift + beta[0]]*len(plot_x), label = r'Shifted $\beta_{}$'.format(i+1), color='orange')
    #     plt.plot(plot_x, [beta[i]]*len(plot_x), label = r'True $\beta_{}$'.format(i+1))
    #     plt.plot(plot_x, beta_fitted[-1, :, i], label="ERM", marker="o", color="red")
    #     for k_ind, k in enumerate(k_candidates):
    #         plt.plot(plot_x, beta_fitted[k_ind,:,i], label = f'k = {k:.1f}', marker=marker_list[k_ind], color=color_list[k_ind])
    #     plt.xlabel(r'log$_{10}\rho$')
    #     plt.ylabel(r'$\beta_{}$'.format(i+1))
    #     plt.legend()
    #     plt.title(f"Beta {i+1}")
    #     if is_save_plot:
    #         plt.savefig(f'plots/{situation}_beta{i+1}.png', bbox_inches='tight')
    #     if is_show_plot:
    #         plt.show()
    #     plt.close()

    plt.figure()
    plt.plot(plot_x, major_loss_mean[-1,:], label='ERM', marker='o', color = 'red')
    plt.errorbar(plot_x, major_loss_mean[-1,:], yerr=major_loss_std[-1,:], capsize=5, color = 'red')
    plt.plot(plot_x, minor_loss_mean[-1,:], marker='o', color = 'red', linestyle = '--')
    plt.errorbar(plot_x, minor_loss_mean[-1,:], yerr=minor_loss_std[-1,:], capsize=5, color = 'red')
    for k_ind, k in enumerate(k_candidates):
        plt.plot(plot_x, major_loss_mean[k_ind,:], label = f'k = {k:.1f}', marker=marker_list[k_ind], color = color_list[k_ind])
        plt.errorbar(plot_x, major_loss_mean[k_ind,:], yerr=major_loss_std[k_ind,:], capsize=5, color = color_list[k_ind])
        plt.plot(plot_x, minor_loss_mean[k_ind,:], marker=marker_list[k_ind], color = color_list[k_ind], linestyle = '--')
        plt.errorbar(plot_x, minor_loss_mean[k_ind,:], yerr=minor_loss_std[k_ind,:], capsize=5, color = color_list[k_ind])
    plt.xlabel(r'log$_{10}\rho$')
    plt.ylabel('average loss')
    plt.legend()
    plt.title("Average Loss")
    if is_save_plot:
        plt.savefig(f'plots/{situation}_major_vs_minor.png', bbox_inches='tight')
    if is_show_plot:
        plt.show()
    plt.show()
    plt.close()