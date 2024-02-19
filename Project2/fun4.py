import pandas as pd
import numpy as np
import itertools as it
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import  MinMaxScaler
from sklearn.model_selection import train_test_split
from cvxopt import matrix, solvers
from sklearn.metrics import accuracy_score
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

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
    primal_objective_value = solution['primal objective']
    alpha = np.array(solution['x']).flatten()
    return alpha


def predict(X_to_pred, alpha, X, Y, gamma, C, tol, ker):
    SV_mask = (alpha > tol) & (alpha < C - tol)
    SV_indices = np.where(SV_mask)[0]
    w_parz = (alpha * Y)
    K_train = K(X, X, gamma, ker)
    K_test = K(X_to_pred, X, gamma, ker)
    b_opt = np.mean((Y - K_train @ w_parz)[SV_indices])
    decision_values = K_test @ w_parz + b_opt
    # y_pred = np.sign(decision_values)
    return decision_values


  ######################################################################################################



def R(alpha, Y, C, tol1=1e-3, tol2=1e-3):
    R = np.where((alpha < C - tol1)*(Y == +1) + (alpha > tol2)*(Y == -1))
    return R


def S(alpha, Y, C, tol1=1e-3, tol2=1e-3):
    S = np.where((alpha < C - tol1)*(Y == -1) + (alpha > tol2)*(Y == +1))
    return S

def select_most_violating_pair(grad, y, R, S,tol):
    m = max((-grad * y)[R])
    M = min((-grad * y)[S])
    delta = m - M

    if delta < tol:
        return None
    item_1 = np.where((-grad * y) == m)[0][0]
    item_2 = np.where((-grad * y) == M)[0][0]
    return item_1, item_2

def calculate_t_max(d, alpha_component, C):
    if d > 1e-8 and alpha_component < C:
        return (C - alpha_component) / d
    elif d < 1e-8 and alpha_component > 1e-8:
        return alpha_component / -d
    #else:
    #    return float('inf')

def compute_optimal_step(grad, Q, alpha, to_change, d, C):
    grad_selected = grad[to_change]
    Q_selected = Q[np.ix_(to_change, to_change)]
    d_Q_d = np.dot(d, Q_selected @ d)
    #se di >0 e alpha < C ok, se di<0 e alpha >0 prendi altro caso, scelgo una cosa che vada bene per entrambe le direzioni
    t_max1 = calculate_t_max(d[0], alpha[to_change][0], C)
    t_max2 = calculate_t_max(d[1], alpha[to_change][1], C)
    t_max_feas = min(t_max1,t_max2)

    if d_Q_d <= 0:
        return t_max_feas #era zero

    t_star = -np.dot(grad_selected, d) / d_Q_d
    
    #max(0, min(t_max_feas, t_star))
    return min(t_max_feas,t_star)

def update_alpha_and_gradient(alpha, grad, Q, to_change, d, t):
    delta_alpha = d * t
    new_alpha = alpha.copy()
    new_alpha[to_change] += delta_alpha

    delta_grad = Q[:, to_change] @ delta_alpha
    new_grad = grad + delta_grad

    return new_alpha, new_grad

def SMO(X, y, ker, gamma, C, tol): 
    P = X.shape[0]
    k = 0  
    Q = np.outer(y, y) * K(X, X, gamma, ker)
    alpha = np.zeros(P)
    grad = -np.ones(P)

    # Inizializzo m e M per entrare nel ciclo while
    m, M = float('inf'), float('-inf')

    while m > M:
        R_set = R(alpha, y, C)  # Aggiorno R
        S_set = S(alpha, y, C)  # Aggiorno S

        pair = select_most_violating_pair(grad, y, R_set, S_set, tol)
        if pair is None:
            break
 
        element_1, element_2 = pair
        to_change = [element_1, element_2]
        d = np.append(y[element_1], -y[element_2])

        t = compute_optimal_step(grad, Q, alpha, to_change, d, C)
        alpha, grad = update_alpha_and_gradient(alpha, grad, Q, to_change, d, t)

        # Aggiorno m e M dopo ogni iterazione
        m = max((-grad * y)[R_set])
        M = min((-grad * y)[S_set])
        k += 1
    print('The KKT violation is:', m-M)
    return alpha


