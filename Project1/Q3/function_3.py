#Group the Treesss
#import the needed libraries
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#we need to make some modification to the original function

#shallow MLP predicted output
def nn_pred(W, v, sigma, N, X): 
    n = X.shape[1] 
    P = X.shape[0]
    W = W.reshape(N, n + 1) #matrice Nx(n+1) 
    v = v.reshape(N, 1) #matrice Nx1

    Z_0 = (np.c_[X, np.ones((P, 1))]).T#X trasposta (nxP), agg riga di 1 ottengo Z_0 spostata alla fine
    Z_1 = np.tanh(sigma*(W @ Z_0))

    Y_pred = np.dot(v.T, Z_1)
    return Y_pred.T 

#simple error function(difference between predicted and true values)
def e(Y_true, X, W, v, sigma, N):
    Y_pred = nn_pred(W, v, sigma, N, X)
    return Y_pred - Y_true 

#regularized error
def E(W, v, sigma, rho, N, Y_true, X): 
    P = X.shape[0]
    omega = np.concatenate((W.flatten(), v.flatten()))
    err = e(Y_true, X, W, v, sigma, N)
    regularaized_error =  0.5*(np.mean(err ** 2) +  rho * np.sum(omega**2)) 
    return  regularaized_error 
 
#gradient of E definition 
def grad_E_omega(W, v, sigma, rho, N, Y_true, X):
    P = X.shape[0]
    n = X.shape[1]
    
    W = W.reshape(N, n + 1)
    v = v.reshape(N, 1)
    omega = np.concatenate((W.flatten(), v.flatten()))
    
    Z_0 = (np.c_[X, np.ones((P, 1))]).T
    Z_1 = np.tanh(sigma * (W @ Z_0)) 
    err = e(Y_true, X, W, v, sigma, N)
    
    grad_v = Z_1 @ err
    grad_W = sigma * (Z_0 @ ((err @ v.T) * (1 - Z_1 ** 2).T))
    #res
    grad_E = np.hstack((grad_W.flatten(order='F'), grad_v.flatten()))
    grad_tot = grad_E / P + rho * omega
    return grad_tot

#gradient of E respect only to W weights
def grad_E_W(W, v, sigma, rho, N, Y_true, X):
    P = X.shape[0]
    n = X.shape[1]
    
    W = W.reshape(N, n + 1)
    v = v.reshape(N, 1)
    
    Z_0 = (np.c_[X, np.ones((P, 1))]).T
    Z_1 = np.tanh(sigma * (W @ Z_0)) 
    err = e(Y_true, X, W, v, sigma, N)
    
    grad_W = sigma * (Z_0 @ ((err @ v.T) * (1 - Z_1 ** 2).T)) 
    #res
    grad_E = (grad_W.flatten(order='F'))
    grad_tot = grad_E / P + rho * W.flatten()
    return grad_tot

#two block method definition
def two_blocks(W, v, sigma, rho, N, Y_true, X, iterations):
    P = X.shape[0]
    n = X.shape[1]

    #counters
    k = 0 
    n_f_ev = 0
    n_j_ev = 0

    #initial guess
    W1 = W.reshape(N, n + 1) 
    Z_0 = (np.c_[X, np.ones((P, 1))]).T
    grad_E = grad_E_omega(W, v, sigma, rho, N, Y_true, X) #initial gradient value

    res = {'param': [], 'error': []} #save results
    
    while np.linalg.norm(grad_E) > 0.5*1e-3 and k < iterations: #check both

        #block one, optimization w.r.t. external weights v --> convex and easy part 
        W1 = W1.reshape(N, n + 1)
        Z_1 = np.tanh(sigma * (W1 @ Z_0))

        v_opt = np.linalg.lstsq((1/P * (Z_1 @ Z_1.T) + rho * np.identity(N)), (1/P * (Z_1 @ Y_true)), rcond=-1)[0]
        #second block, non convex optimization w.r.t. inner weights W
        result = minimize(E, W1, args = (v_opt, sigma, rho, N, Y_true, X), jac = grad_E_W, options={'gtol':1e-4})
        n_f_ev += result.nfev
        n_j_ev += result.njev

        W1 = result.x
        omega_new = np.hstack((W1.flatten(), v_opt.flatten()))
        #mse error
        E_current_2 = 0.5 * mean_squared_error(Y_true, nn_pred(W1, v_opt, sigma, N, X))# + rho * np.sum(omega_new**2)
        grad_E = grad_E_omega(W1, v_opt, sigma, rho, N, Y_true, X)

        k += 1

        res['param'].append(omega_new)
        res['error'].append(E_current_2)

    return pd.DataFrame(res), [k, n_f_ev, n_j_ev]

##########################################################


def nn_pred2(omega, sigma, N, X): 
    n = X.shape[1] #2
    P = X.shape[0]
    W = (omega[: (N * (n + 1))]).reshape(N, n + 1) #matrix Nx(n+1)
    v = (omega[(N * (n + 1)):]).reshape(N, 1) #matrix Nx1

    Z_0 = (np.c_[X, np.ones((P, 1))]).T#X.T is (nxP), add 1 row of ones to have Z_0 
    Z_1 = np.tanh(sigma*(W @ Z_0))
    Y_pred = np.dot(v.T, Z_1)
    return Y_pred.T 

def e2(Y_true, X, omega, sigma, N):
    Y_pred = nn_pred2(omega, sigma, N, X)
    return Y_pred - Y_true  

#gradient of E definition
def grad_E_omega2(omega, sigma, rho, N, Y_true, X):
    P = X.shape[0]
    n = X.shape[1]
    
    W = omega[:N * (n + 1)].reshape(N, n + 1)
    v = omega[N * (n + 1):].reshape(N, 1)
    
    Z_0 = (np.c_[X, np.ones((P, 1))]).T
    Z_1 = np.tanh(sigma * (W @ Z_0)) 
    err = e2(Y_true, X, omega, sigma, N)
    
    grad_v = Z_1 @ err
    grad_W = sigma * (Z_0 @ ((err @ v.T) * (1 - Z_1 ** 2).T)) 
    #res
    grad_E = np.hstack((grad_W.flatten(order='F'), grad_v.flatten()))
    grad_tot = grad_E / P + rho * omega
    return grad_tot

#predicted function plot
def fun_plot3(omega, sigma, N):
    x1_vals = np.linspace(-2, 2, 100)
    x2_vals = np.linspace(-3, 3, 100)
    x1, x2 = np.meshgrid(x1_vals, x2_vals)
    X_grid = np.column_stack((x1.ravel(), x2.ravel()))

    n = X_grid.shape[1]
    W = omega[:N * (n + 1)].reshape(N, n + 1)
    v = omega[N * (n + 1):].reshape(N, 1)

    y_hat = nn_pred(W, v, sigma, N, X_grid)
    y_hat = y_hat.reshape(x1.shape)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    #plot of the continuous surface of the function
    surf = ax.plot_surface(x1, x2, y_hat, cmap='viridis')
    ax.set_xlabel('Feature x1')
    ax.set_ylabel('Feature x2')
    ax.set_zlabel('Predicted output')
    ax.set_title('Surface Plot of the Shallow MLP Neural Network prediction')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()