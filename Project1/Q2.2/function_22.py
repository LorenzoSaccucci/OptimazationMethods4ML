#Group the Treesss
#import of needed libraries
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import random


#####################################################################
#functions definition:

#Shallow RBF network train and prediction
def RBF(W,X,N,sigma):
    n = X.shape[1]

    C = np.reshape(W[:N*n],(N,n))
    v = W[N*n:]

    #subtraction of each vector of coefficients from each feature sample using data broadcasting
    diff = X[:, np.newaxis, :] - C  #compute differences between X and C
    y_pred = np.dot(np.exp(-(1 / (sigma**2)) * np.linalg.norm(diff, axis=2)**2), v)
    
    return y_pred


#random indeces identification in order to select randomly the centers among features samples
def sample_rows(X, N,seed):
    np.random.seed(seed)
    indices = np.random.choice(len(X), N, replace=False)
    return X[indices]


#unsupervised selection of centers
def unsupervised_centers_selection(X,Y, N, sigma,rho,seed):
    P = X.shape[0]

    C = sample_rows(X, N,seed)
    #phi matrix, gaussian functions applied to squared norm differences among all centers and all features
    PHI = np.exp(-(1 / (sigma**2)) * np.linalg.norm(X[:, np.newaxis, :] - C, axis=2)**2) 

    #optimize only weights v
    v_opt = np.linalg.lstsq(( 1/P * (PHI.T @ PHI) + rho * np.identity(N)), (1/P * (PHI.T @ Y)), rcond=-1)[0]
    W = np.append(C.flatten(),v_opt) #tot weights

    E = 0.5* mean_squared_error(Y, RBF(W,X,N,sigma))+rho * np.linalg.norm(v_opt, axis=0)**2
    return E,W


#####################################################################

#multistart definition
#def multistart_procedure(X, Y, N, sigma, rho, iterations):
#    E_extreme = 1e10
#    omega_star = None
#    best_seed = None
#    for i in range(iterations):
#        E_current, W= centers_selection(X, Y, N, sigma, rho, i)
#        if E_current < E_extreme:
#            E_extreme = E_current
#            omega_star = W
#            best_seed = i

#    return E_extreme, omega_star, best_seed


######################################################################


def Plot_function(W, N, sigma):
    x_1 = np.linspace(-2, 2, 1000)
    x_2 = np.linspace(-3, 3, 1000)
    X_1, X_2 = np.meshgrid(x_1, x_2)
    in_1 = np.array([])
    in_2 = np.array([])
    in_1 = X_1.flatten()
    in_2 = X_2.flatten()
    points = np.c_[in_1, in_2]
    Y = RBF(W, points, N, sigma) 
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    surf=ax.plot_surface(X_1, X_2, Y.reshape(1000, 1000), cmap='viridis')
    ax.set_xlabel('Feature x1')
    ax.set_ylabel('Feature x2')
    ax.set_zlabel('Predicted output')
    ax.set_title('Surface Plot of the Shallow RBF Neural Network prediction')
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

    return
