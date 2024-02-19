#Group the Treesss
#import of needed libraries
import numpy as np
import pandas as pd
import itertools
from scipy.optimize import minimize
from sklearn.model_selection import KFold #?
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


##############################
#functions definition:

#shallow MLP predicted output
def nn_pred(omega, sigma, N, X): 
    n = X.shape[1] #2
    P = X.shape[0]
    W = (omega[: (N * (n + 1))]).reshape(N, n + 1) #matrix Nx(n+1)
    v = (omega[(N * (n + 1)):]).reshape(N, 1) #matrix Nx1

    Z_0 = (np.c_[X, np.ones((P, 1))]).T#X.T is (nxP), add 1 row of ones to have Z_0 
    Z_1 = np.tanh(sigma*(W @ Z_0))
    Y_pred = np.dot(v.T, Z_1)
    return Y_pred.T #transposition to have column vector

#simple error function(difference between predicted and true values)
def e(Y_true, X, omega, sigma, N):
    Y_pred = nn_pred(omega, sigma, N, X)
    return Y_pred - Y_true 

#regularized error
def E(omega, sigma, rho, N, Y_true, X): 
    P = X.shape[0]
    err = e(Y_true, X, omega, sigma, N)
    regularaized_error =  0.5*((1/P)*np.linalg.norm(err)**2 +  rho * np.linalg.norm(omega)**2) #modificato con (1/P)*np.linalg.norm(err)**2
    return  regularaized_error 


#gradient of E definition
def grad_E_omega(omega, sigma, rho, N, Y_true, X):
    P = X.shape[0]
    n = X.shape[1]
    
    W = omega[:N * (n + 1)].reshape(N, n + 1)
    v = omega[N * (n + 1):].reshape(N, 1)
    
    Z_0 = (np.c_[X, np.ones((P, 1))]).T
    Z_1 = np.tanh(sigma * (W @ Z_0)) 
    err = e(Y_true, X, omega, sigma, N)
    
    grad_v = Z_1 @ err
    grad_W = sigma * (Z_0 @ ((err @ v.T) * (1 - Z_1 ** 2).T)) 
    #res
    grad_E = np.hstack((grad_W.flatten(order='F'), grad_v.flatten()))
    grad_tot = grad_E / P + rho * omega
    return grad_tot



#########################
#function plot:
#plot dei punti con colore in base alle predizioni y_hat
def fun_plot1(omega, sigma, N, X):
    y_hat = nn_pred(omega, sigma, N, X) 

    #plot delle predizioni
    plt.figure(figsize=(8, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y_hat, cmap='viridis')  
    plt.colorbar()  
    plt.xlabel('Feature x1')
    plt.ylabel('Feature x2')
    plt.title('Shallow Neural Network Predictions with sigma = {}'.format(sigma))
    plt.show()

#just dots
def fun_plot2(omega, sigma, N, X):
    y_hat = nn_pred(omega, sigma, N, X)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    #prediction plot in a 3D graph
    ax.scatter(X[:, 0], X[:, 1], y_hat, c=y_hat, cmap='viridis')
    ax.set_xlabel('Feature x1')
    ax.set_ylabel('Feature x2')
    ax.set_zlabel('Predicted output')
    ax.set_title('Shallow Neural Network Predictions with sigma = {}'.format(sigma))

    plt.show()

#predicted function plot
def fun_plot3(omega, sigma, N):
    x1_vals = np.linspace(-2, 2, 100)
    x2_vals = np.linspace(-3, 3, 100)
    x1, x2 = np.meshgrid(x1_vals, x2_vals)
    X_grid = np.column_stack((x1.ravel(), x2.ravel()))

    y_hat = nn_pred(omega, sigma, N, X_grid)
    y_hat = y_hat.reshape(x1.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    #plot of the continuous surface of the function
    surf = ax.plot_surface(x1, x2, y_hat, cmap='viridis')
    ax.set_xlabel('Feature x1')
    ax.set_ylabel('Feature x2')
    ax.set_zlabel('Predicted output')
    ax.set_title('Surface Plot of the Shallow Neural Network prediction')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

############################################################################

#grid_search function
def grid_search_custom_MLP(X_train, y_train, possible_N, possible_sigma, possible_rho, K, random_seed):
    best_params = None
    best_val_score = float('inf')
    
    results = pd.DataFrame(columns=['N', 'sigma', 'rho', 'train_score', 'val_score', 'seed'])#save results

    data_training = np.column_stack((X_train, y_train))
    np.random.shuffle(data_training)

    folds = np.array_split(data_training, K) #create the different folds

    #iteration over all possible values
    for N, sigma, rho in itertools.product(possible_N, possible_sigma, possible_rho):
        train_score = 0
        val_score = 0

        for k in range(K):
            k_th_folds = folds.copy()
            test_fold = k_th_folds.pop(k)
            train_folds = np.concatenate(k_th_folds)
            np.random.seed(random_seed) #pass the random seed
            initial_weights = np.random.random(N * 4)  
            res = minimize(E, initial_weights, args=(sigma, rho, N, train_folds[:, -1].reshape(-1, 1), train_folds[:, :-1]), jac=grad_E_omega, method='BFGS', options={'gtol': 0.5 * 1e-5})
            val_score += (0.5 / K) * mean_squared_error(test_fold[:, -1], nn_pred(res.x, sigma, N, test_fold[:, :-1]))
            train_score += (0.5 / K) * mean_squared_error(train_folds[:, -1], nn_pred(res.x, sigma, N, train_folds[:, :-1]))

        results.loc[len(results.index)] = [N, sigma, rho, train_score, val_score, random_seed]
        #update
        if val_score < best_val_score:
            best_params = (N, sigma, rho)
            best_val_score = val_score

    return results, best_params

#this is need in order to try different seeds, from the very beginning
def multistart_grid_search_custom_MLP(X_train, y_train, possible_N, possible_sigma, possible_rho, K, num_starts=50):
    best_params_overall = None
    best_val_score_overall = float('inf')
    best_seed_overall = None
    all_results = pd.DataFrame()  

    for i in range(num_starts):
        random_seed = np.random.randint(1, 10000)#create a random value
        #recall the 'standard' grid_search
        results, best_params = grid_search_custom_MLP(X_train, y_train, possible_N, possible_sigma, possible_rho, K, random_seed)
        all_results = pd.concat([all_results, results])  #concatenate dataframe

        if results['val_score'].min() < best_val_score_overall:
            best_params_overall = best_params
            best_val_score_overall = results['val_score'].min()
            best_seed_overall = random_seed

    return all_results, best_params_overall, best_seed_overall

