#Group the Treesss
#import of needed libraries
import numpy as np
import pandas as pd
import itertools
from scipy.optimize import minimize
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#gaussian function 
def gauss(x, sigma):
    return np.exp(-(x/sigma)**2)

#matrix phi, gaussian computed in the squared norm of differences (between each center and each feature)
def phi(omega, N, sigma, X):
    n = X.shape[1]
    c = (omega[N:].reshape(N, n)) 
    #subtraction of each vector of coefficients from each feature sample using data broadcasting
    diff = X[:,np.newaxis, :] - c  
    #norm function applied to each differences along axis n+1 
    norms = np.linalg.norm(diff, axis=2) 
    
    return gauss(norms, sigma)

#shallow radial basis function network prediction
def rbf_pred(omega, N, sigma, X):
    v = omega[:N].reshape(N,1)
    return phi(omega, N, sigma, X) @ v
  
#regularized error functions

def E2(omega, N, sigma, rho, X, Y_true): #faster version
    rho1, rho2 = rho, rho
    P = X.shape[0]
    n = X.shape[1]

    #params
    v = omega[:N].reshape(N,1)
    c = (omega[N:].reshape(N, n))

    Y_pred = rbf_pred(omega, N, sigma, X)
    e = Y_pred - Y_true
    regularized_error = 0.5 * (1/P) * ((np.linalg.norm(e))**2 + P * (rho1 * np.linalg.norm(v)**2 + rho2 * np.linalg.norm(c)**2))

    return regularized_error

def E3(omega, N, sigma, rho, X, Y_true): 
    P = X.shape[0]
    n = X.shape[1]

    #params
    rho1, rho2 = rho, rho #anziché rho1, rho2 = rho[0], rho[1]
    v = omega[:N].reshape(N,1)
    c = (omega[N:].reshape(N, n))

    #build first part
    matrix_phi = phi(omega, N, sigma, X)
    matrix_I = np.eye(N)*np.sqrt(P*rho1)
    matrix = np.vstack((matrix_phi, matrix_I))
    #build second part
    column_of_zeros = np.zeros((N, 1))
    vector = np.vstack((Y_true, column_of_zeros))

    regularaized_error =  0.5 * (1/P)* (np.sum((matrix @ v - vector)**2) + P * rho2 * (np.sum(c**2)))

    return regularaized_error

def grad_E_rbf(omega, N, sigma, rho, X, Y_true):
    P = X.shape[0]
    n = X.shape[1]

    #params
    rho1, rho2 = rho, rho
    c = (omega[N:].reshape(N, n))
    v = omega[:N].reshape(N,1)

    e = rbf_pred(omega, N, sigma, X) - Y_true
    matrix_phi = phi(omega, N, sigma, X)

    #grad_v
    #grad_v = (((matrix_phi.T @ matrix_phi) + np.eye(N)*P*rho1)@v - matrix_phi.T @ Y_true).flatten() 
    grad_v = (1/P)*((matrix_phi.T @ (matrix_phi @ v - Y_true)) + P*rho1*v).flatten()#equivalent
    #grad_c:
    diff = X[:,np.newaxis, :] - c
    result = matrix_phi[:, :, np.newaxis] * diff
    scalar_product = np.tensordot(e.T, result, axes=([1], [0])) #matrix N x n
    grad_c = (1/P)*((v * scalar_product)*(2/sigma**2)).flatten() + (rho2 * c.flatten())
        
    grad_tot = np.hstack((grad_v, grad_c)) 

    return grad_tot
        
############################################################
        
#predicted function plot
def fun_plot3(omega, sigma, N):
    x1_vals = np.linspace(-2, 2, 100)
    x2_vals = np.linspace(-3, 3, 100)
    x1, x2 = np.meshgrid(x1_vals, x2_vals)
    X_grid = np.column_stack((x1.ravel(), x2.ravel()))

    y_hat = rbf_pred(omega, N, sigma, X_grid)
    y_hat = y_hat.reshape(x1.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    #plot della superficie continua della funzione
    surf = ax.plot_surface(x1, x2, y_hat, cmap='viridis')
    ax.set_xlabel('Feature x1')
    ax.set_ylabel('Feature x2')
    ax.set_zlabel('Predicted output')
    ax.set_title('Surface Plot of the Shallow RBF network prediction')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()
        
#############################################################

#grid_search function
def grid_search_custom_RBF(X_train, y_train, possible_N, possible_sigma, possible_rho, K, random_seed):
    n = X_train.shape[1] 
    best_params = None
    best_val_score = float('inf')
    
    results = pd.DataFrame(columns=['N', 'sigma', 'rho', 'train_score', 'val_score', 'seed']) #save results

    data_training = np.column_stack((X_train, y_train))
    np.random.shuffle(data_training)

    folds = np.array_split(data_training, K)#create the different folds

    #iteration over all possible values
    for N, sigma, rho in itertools.product(possible_N, possible_sigma, possible_rho):
        train_score = 0
        val_score = 0

        for k in range(K):
            k_th_folds = folds.copy()
            test_fold = k_th_folds.pop(k)
            train_folds = np.concatenate(k_th_folds)
            np.random.seed(random_seed)  #pass the random seed  
            initial_weights = np.random.random(N * (n+1)) 
            res = minimize(E2, initial_weights, args=(N, sigma, rho, train_folds[:, :-1], train_folds[:, -1].reshape(-1, 1)), jac=grad_E_rbf, method='BFGS', options={'gtol': 0.5 * 1e-5})
            val_score += (0.5 / K) * mean_squared_error(test_fold[:, -1], rbf_pred(res.x, N, sigma, test_fold[:, :-1]))
            train_score += (0.5 / K) * mean_squared_error(train_folds[:, -1], rbf_pred(res.x, N, sigma, train_folds[:, :-1]))

        results.loc[len(results.index)] = [N, sigma, rho, train_score, val_score, random_seed]
        #update
        if val_score < best_val_score:
            best_params = (N, sigma, rho)
            best_val_score = val_score

    return results, best_params

#this is need in order to try different seeds, from the very beginning
def multistart_grid_search_custom_RBF(X_train, y_train, possible_N, possible_sigma, possible_rho, K, num_starts=50):
    best_params_overall = None
    best_val_score_overall = float('inf')
    best_seed_overall = None
    all_results = pd.DataFrame()  

    for i in range(num_starts):
        random_seed = np.random.randint(1, 10000)#create a random value
        #recall the 'standard' grid_search
        results, best_params = grid_search_custom_RBF(X_train, y_train, possible_N, possible_sigma, possible_rho, K, random_seed)
        all_results = pd.concat([all_results, results])  #concatenate dataframe

        if results['val_score'].min() < best_val_score_overall:
            best_params_overall = best_params
            best_val_score_overall = results['val_score'].min()
            best_seed_overall = random_seed

    return all_results, best_params_overall, best_seed_overall

