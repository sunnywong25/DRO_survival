import numpy as np
import json
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from torch.autograd import Variable 
from dro_models import *
from numpy.random import normal, binomial

#The function below ensures that we seed all random generators with the same value to get reproducible results
def set_random_seed(state=1):
    gens = (np.random.seed, torch.manual_seed, torch.cuda.manual_seed)
    for set_state in gens:
        set_state(state)

def simulate_linear_subpop_dgp(beta_major, beta_minor, sigma = 0.1, sim_n=200, G_dist = lambda n: binomial(1, 0.9, n)):
    sim_X = normal(size=(sim_n, len(beta_major)))
    G = G_dist(sim_n)
    pred = G*np.dot(sim_X, beta_major) + (1 - G)*np.dot(sim_X, beta_minor)
    sim_y = normal(loc=pred, scale=sigma).reshape((-1,1))
    return sim_X, sim_y, G

if __name__ == "__main__":
    with open("C:\\Users\\Administrator\\Desktop\\DRO_survival\\dro_simulation\\dro_subpop_simulation_config.json", "r") as f:
        sim_config = json.load(f)
    print("Simulation config:", sim_config)
    is_show_plot = True
    situation = sim_config['situation']
    marker_list = ['v', 's', '*']
    color_list = ['green', 'blue', 'black']

    eval_test = lambda y, pred: 0.5*(y - pred)**2
    sq_err_indiv = lambda y, pred: (y - pred)**2
    true_test_loss = lambda beta_fitted, beta_true, sigma: 0.5 * (np.linalg.norm(beta_fitted.reshape(-1,) - beta_true)**2 + sigma**2)
    
    # Simulation config
    dump_pickle_path = sim_config['dump_pickle_path']
    random_state = sim_config['random_state']
    sim_n_runs = sim_config['sim_n_runs']
    sim_n_train = sim_config['sim_n_train']
    sim_n_test = sim_config['sim_n_test']
    beta_major = np.array(sim_config['beta_major'])
    beta_minor = np.array(sim_config['beta_minor'])
    sigma = sim_config['sigma']
    minor_prop = sim_config['minor_prop']
    G_dist = lambda n: binomial(1, 1 - minor_prop, n)
    #G_dist = lambda n: np.random.beta(1.0, 2.0/3.0, n)

    # Grids
    k_candidates = sim_config['k_candidates']
    rho_candidates = 10**np.array(sim_config['log10_rho_candidates'])

    # Model training parameters
    bias = sim_config['bias']
    epochs = sim_config['epochs'] # max epochs
    lr = sim_config['lr']
    verbose = sim_config['verbose']

    # Initializations
    set_random_seed(random_state)
    avg_test_loss = np.zeros((sim_n_runs, len(k_candidates)+1, len(rho_candidates)))
    loss_on_minor = np.zeros((sim_n_runs, len(k_candidates)+1, len(rho_candidates)))
    beta_fitted = np.zeros((sim_n_runs, len(k_candidates)+1, len(rho_candidates), len(beta_major)))
    eta_fitted = np.zeros((sim_n_runs, len(k_candidates), len(rho_candidates)))

    for run_ind in range(sim_n_runs):
        print(f"Run {run_ind + 1}/{sim_n_runs}")
        # Simulation
        # Generate datasets
        sim_X_train, sim_y_train, G_train = simulate_linear_subpop_dgp(beta_major, beta_minor, sigma=sigma, sim_n=sim_n_train, G_dist=G_dist)
        sim_X_test, sim_y_test, G_test = simulate_linear_subpop_dgp(beta_major, beta_minor, sigma=sigma, sim_n=sim_n_test, G_dist=G_dist)

        # Convert np.array to torch.tensor if using torch implementation
        sim_X_train_tensor = Variable((torch.from_numpy(sim_X_train)).float())
        sim_y_train_tensor = Variable((torch.from_numpy(sim_y_train)).float())
        sim_X_test = torch.from_numpy(sim_X_test).float()

        # ERM baseline
        model = LinearModel(len(beta_major), 1, bias)
        loss_indiv = nn.MSELoss(reduction="none")
        fit_erm(model=model, 
                X_tensor=sim_X_train_tensor, 
                list_of_targets=[sim_y_train_tensor], 
                loss_indiv=loss_indiv, 
                epochs=epochs,
                lr=lr,
                verbose=verbose)
        pred_test = model(sim_X_test).detach().numpy().reshape((-1,))
        test_loss_indiv = eval_test(sim_y_test.reshape(-1,), pred_test)
        avg_test_loss[run_ind, -1, :] = test_loss_indiv.mean()
        loss_on_minor[run_ind, -1, :] = test_loss_indiv[G_test==0].mean()
        beta_fitted[run_ind,-1, :, :] = [para.detach().numpy() for para in model.parameters()][0]

        for k_ind, k in enumerate(k_candidates):
            for rho_ind, rho in enumerate(rho_candidates):
                print(f"{rho:.2f}:", end = '')

                # torch implementation
                model = LinearModel(len(beta_major), 1, bias)
                loss_indiv = nn.MSELoss(reduction="none")
                dro_loss = DroLoss(loss_indiv, div_family_k=k, radius_rho=rho)
                dro_loss_trace = fit_dro(model=model, 
                                        X_tensor=sim_X_train_tensor, 
                                        list_of_targets=[sim_y_train_tensor], 
                                        dro_loss=dro_loss, 
                                        epochs=epochs,
                                        lr=lr,
                                        verbose=verbose)
                beta_fitted[run_ind,k_ind, rho_ind, :] = [para.detach().numpy() for para in model.parameters()][0]
                eta_fitted[run_ind,k_ind, rho_ind] = [para.detach().numpy() for para in dro_loss.parameters()][0]
                pred_test = model(sim_X_test).detach().numpy().reshape((-1,))
                test_loss_indiv = eval_test(sim_y_test.reshape(-1,), pred_test)
                avg_test_loss[run_ind,k_ind, rho_ind] = test_loss_indiv.mean()
                loss_on_minor[run_ind,k_ind, rho_ind] = test_loss_indiv[G_test==0].mean()
    
    for name in ['avg_test_loss', 'loss_on_minor', 'beta_fitted', 'eta_fitted']:
        dump_pickle(eval(name), f'{dump_pickle_path}/{situation}_{name}.pkl')
        print(f"Dumped {name} to data/situation_{name}.pkl")