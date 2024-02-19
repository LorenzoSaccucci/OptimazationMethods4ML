#Group the Treesss
#import of needed libraries
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random

#####################################################################
#functions definition:

#Shallow MLP network train and prediction
def f_of_x(weights, X, N, sig):
    P = X.shape[0]
    n = X.shape[1]

    w = weights[:N*(n+1)].reshape(N, n+1)
    v = weights[N*(n+1):]

    Z = np.c_[X, np.ones(P)]

    y_pred = np.tanh(sig * (Z @ w.T)) @ v
    return y_pred

#definition of extreme learning method
def Extreme_Learning(X, Y, N, sigma, rho, seed):
    P = X.shape[0]
    n = X.shape[1]

    np.random.seed(seed)
    w_random = np.random.uniform(-2.0, 2.0, N * 3).reshape(N, n + 1) 

    Z = np.c_[X, np.ones(P)]
    Q = np.tanh(sigma * (Z @ w_random.T))

    #optimize only weights v
    v_opt = np.linalg.lstsq((1/P * (Q.T @ Q) + rho * np.identity(N)), (1/P * (Q.T @ Y)), rcond=-1)[0]
    W = np.append(w_random.flatten(), v_opt) #tot weights

    E = 0.5 * mean_squared_error(Y, f_of_x(W, X, N, sigma)) + rho * np.linalg.norm(v_opt, axis=0)**2 #regularized error
    return E, W



#######################################################################

#multistart definition
# def multistart_procedure(X, Y, N, sigma, rho, iterations):
#     E_extreme = 1e10
#     omega_star = None
#     best_seed = None
#     for i in range(iterations):
#         E_current, W= Extreme_Learning(X, Y, N, sigma, rho, i)
#         if E_current < E_extreme:
#             E_extreme = E_current
#             omega_star = W
#             best_seed = i

#     return E_extreme, omega_star, best_seed


######################################################


def Plot_function(W, N, sigma):

    x_1 = np.linspace(-2, 2, 1000)
    x_2 = np.linspace(-3, 3, 1000)
    X_1, X_2 = np.meshgrid(x_1, x_2)
    in_1 = np.array([])
    in_2 = np.array([])
    in_1 = X_1.flatten()
    in_2 = X_2.flatten()
    points = np.c_[in_1, in_2]
    Y = f_of_x(W, points, N, sigma) 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf=ax.plot_surface(X_1, X_2, Y.reshape(1000, 1000), cmap='viridis')
    ax.set_xlabel('Feature x1')
    ax.set_ylabel('Feature x2')
    ax.set_zlabel('Predicted output')
    ax.set_title('Surface Plot of the Shallow MLP Neural Network prediction')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    return
