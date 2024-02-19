import pandas as pd
import numpy as np
import itertools as it
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  MinMaxScaler
from cvxopt import matrix, solvers
from sklearn.metrics import accuracy_score
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from matplotlib.pyplot import imshow

def map_labels(label):
    if label == 2:
        return 1
    elif label == 3:
        return -1

# Defining the kernel functions for two given matrices X and Y
def K(X, Y, gamma, kernel):
    if kernel == 'Polynomial':
        return (1 + X @ Y.T)**gamma
    elif kernel == 'Gaussian':
        norms = np.linalg.norm(X, axis=1)**2
        norms_Y = np.linalg.norm(Y, axis=1)**2
        centr = -2 * (X @ Y.T)
        return np.exp(-gamma * (norms[:, np.newaxis] + centr + norms_Y[np.newaxis, :]))
        

def alpha_opt(X, y, gamma, C, ker): 
    k = K(X, X, gamma, kernel=ker)
    Q = k * np.outer(y, y) 
    P = X.shape[0]
    q = -np.ones(P) 
    G = np.vstack([-np.eye(P), np.eye(P)])
    h = np.concatenate([-np.zeros(P), np.ones(P) * C])
    A = y.reshape(1, P).astype('float') 
    b = np.array([0.])
    solution = solvers.qp(P=matrix(Q), q=matrix(q), G=matrix(G),h=matrix(h),A=matrix(A), b=matrix(b,tc='d'))
    alpha = np.array(solution['x']).flatten()
    return alpha


def decision_function(X_to_pred, alpha, X, Y, gamma, C, tol, ker):
    SV_mask = (alpha > tol) & (alpha < C - tol)
    SV_indices = np.where(SV_mask)[0]
    w_parz = (alpha * Y)
    K_train = K(X, X, gamma, ker)
    K_test = K(X_to_pred, X, gamma, ker)
    b_opt = np.mean((Y - K_train @ w_parz)[SV_indices])
    decision_values = K_test @ w_parz + b_opt
    y_pred = np.sign(decision_values)
    return y_pred


def R(alpha, Y, C, tol1, tol2):
    R = np.where((alpha < C - tol1)*(Y == +1) + (alpha > tol2)*(Y == -1))
    return R


def S(alpha, Y, C, tol1, tol2):
    S = np.where((alpha < C - tol1)*(Y == -1) + (alpha > tol2)*(Y == +1))
    return S

def obj_func(X,y, alpha, gamma, ker):
    P = X.shape[0]
    k = K(X,X,gamma, ker )
    Q = k * np.outer(y,y)
    fo = 0.5*np.dot(alpha, Q.T @ alpha) - np.ones(P)@alpha
    return  fo



# def grid_search_custom_SVM_G(X_train, y_train, possible_C, possible_gamma, K, random_seed, ker ='Gaussian'):
#     best_params = None
#     best_val_accuracy = 0.0

#     results = pd.DataFrame(columns=['C', 'gamma', 'train_accuracy', 'val_accuracy', 'seed'])

#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_seed)

#     # Iteration over all possible values
#     for C, gamma in it.product(possible_C, possible_gamma):
#         train_accuracy = 0.0
#         val_accuracy = 0.0

#         for k in range(K):
#             alpha = alpha_opt(X_train, y_train, gamma, C, ker='Gaussian')
#             y_train_pred = decision_function(X_train, alpha, X_train, y_train, gamma, C, tol=1e-4, ker='Gaussian')
#             y_val_pred = decision_function(X_val, alpha, X_train, y_train, gamma, C, tol=1e-4, ker='Gaussian')

#             train_accuracy += (1.0 / K) * accuracy_score(y_train, y_train_pred)
#             val_accuracy += (1.0 / K) * accuracy_score(y_val, y_val_pred)

#         results.loc[len(results.index)] = [C, gamma, train_accuracy, val_accuracy, random_seed]

#         # Update
#         if val_accuracy > best_val_accuracy:
#             best_params = (C, gamma)
#             best_val_accuracy = val_accuracy

#     results.to_excel("grid_search_results1.xlsx")

#     return results, best_params

# def grid_search_custom_SVM_P(X_train, y_train, possible_C, possible_gamma, K, random_seed, ker='Polynomial'):
#     best_params = None
#     best_val_accuracy = 0.0

#     results = pd.DataFrame(columns=['C', 'gamma', 'train_accuracy', 'val_accuracy', 'seed', 'error'])

#     X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_seed)

#     # Iteration over all possible values
#     for C, gamma in it.product(possible_C, possible_gamma):
#         try:
#             train_accuracy = 0.0
#             val_accuracy = 0.0

#             for k in range(K):
#                 alpha = alpha_opt(X_train, y_train, gamma, C, ker='Polynomial')
#                 y_train_pred = decision_function(X_train, alpha, X_train, y_train, gamma, C, tol=1e-4, ker='Polynomial')
#                 y_val_pred = decision_function(X_val, alpha, X_train, y_train, gamma, C, tol=1e-4, ker='Polynomial')

#                 train_accuracy += (1.0 / K) * accuracy_score(y_train, y_train_pred)
#                 val_accuracy += (1.0 / K) * accuracy_score(y_val, y_val_pred)

#             results.loc[len(results.index)] = [C, gamma, train_accuracy, val_accuracy, random_seed, None]

#             # Update best parameters
#             if val_accuracy > best_val_accuracy:
#                 best_params = (C, gamma)
#                 best_val_accuracy = val_accuracy

#         except Exception as e:
#             # Log the error
#             results.loc[len(results.index)] = [C, gamma, None, None, random_seed, str(e)]

#     results.to_excel("grid_search_results_poly.xlsx")
#     return best_params, best_val_accuracy