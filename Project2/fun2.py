
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from cvxopt import matrix, solvers
import itertools as it
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

def map_labels(label):
    if label == 2:
        return 1
    elif label == 3:
        return -1

def K(X, Y, gamma, kernel):
    if kernel == 'Polynomial':
        return (1 + X @ Y.T)**gamma
    elif kernel == 'Gaussian':
        norms = np.linalg.norm(X, axis=1)**2
        norms_Y = np.linalg.norm(Y, axis=1)**2
        centr = -2 * (X @ Y.T)
        return np.exp(-gamma * (norms[:, np.newaxis] + centr + norms_Y[np.newaxis, :]))

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
    
def seek_R(alpha, Y, C, tol1, tol2):
    R = np.where((alpha < C - tol1)*(Y == +1) + (alpha > tol2)*(Y == -1))
    return R

def seek_S(alpha, Y, C, tol1, tol2):
    S = np.where((alpha < C - tol1)*(Y == -1) + (alpha > tol2)*(Y == +1))
    return S

#evaluate the KKT conditions

#fuction to identify the working set

def select_W(grad, y, R, S, q):
    grad_f = -grad * y 
    R, S = R[0], S[0]
    W = []

    for _ in range(q // 2):
        max_value = np.max(grad_f[R])
        max_index = np.where(grad_f[R] == max_value)[0][0]
        W.append(R[max_index])
        grad_f[R[max_index]] = -np.inf  #set the value to -inf to ignore it in the next iterations

    grad_f = -grad * y

    for _ in range(q // 2):
        min_value = np.min(grad_f[S])
        min_index = np.where(grad_f[S] == min_value)[0][0]
        W.append(S[min_index])
        grad_f[S[min_index]] = np.inf  #set the value to inf to ignore it in the next iterations
    W = np.unique(np.array(W))
    return W


# #main function for the decomposition algorithm

def kkt_violation(grad, y, R, S):
    grad_f = -grad * y
    R_indices, S_indices = R[0], S[0]
    m, M = np.max(grad_f[R_indices]), np.min(grad_f[S_indices])
    return m - M

def alpha_solv(Q_matrix, y, alpha_fix, W_k, W_bar, C):
    try:
        Q_WW = Q_matrix[np.ix_(W_k, W_k)]
        Q_WWbar = Q_matrix[np.ix_(W_k, W_bar)]
        Q = Q_WW 
        q = (Q_WWbar @ alpha_fix) - 1
        PW = len(W_k)
        G = np.vstack([-np.eye(PW), np.eye(PW)])
        h = np.concatenate([np.zeros(PW), np.ones(PW) * C])
        A = y[W_k].astype('float').reshape(1, PW)
        b = -y[W_bar].T @ alpha_fix
        solution = solvers.qp(P=matrix(Q), q=matrix(q), G=matrix(G), h=matrix(h), A=matrix(A), b=matrix(b, tc='d'), options={'show_progress':False})
        return np.array(solution['x']).flatten()
    except Exception as e:
        return np.zeros(len(W_k))

def decomposition_alg(X, Z, y, gamma, ker, tol1, tol2, q, C):
    k = 0
    max_iter = 101
    P = X.shape[0]
    Q_matrix = K(X, Z, gamma, ker) * np.outer(y, y)
    alpha = np.zeros(P)
    grad = -np.ones(P)

    R = seek_R(alpha, y, C, tol1, tol2)
    S = seek_S(alpha, y, C, tol1, tol2)
    kkt_values = kkt_violation(grad, y, R, S)

    while k < max_iter:
        if kkt_values < tol1:
            break

        W_k = select_W(grad, y, R, S, q)
        W_bar = np.setdiff1d(np.arange(P), W_k)

        alpha_fix = alpha[W_bar]
        alpha_old = alpha[W_k]
        
        alpha_opt = alpha_solv(Q_matrix, y, alpha_fix, W_k, W_bar, C)

        grad += Q_matrix[:, W_k] @ (alpha_opt - alpha_old)
        alpha[W_k] = alpha_opt
        
        R = seek_R(alpha, y, C, tol1, tol2)
        S = seek_S(alpha, y, C, tol1, tol2)
        kkt_values = kkt_violation(grad, y, R, S)

        k += 1
    return alpha, kkt_values

def obj_func(X,y, alpha, gamma, ker):
    P = X.shape[0]
    k = K(X,X,gamma, ker )
    Q = k * np.outer(y,y)
    fo = 0.5*np.dot(alpha, Q.T @ alpha) - np.ones(P)@alpha
    return  fo

# def grid_search_custom_SVM(X_train, y_train, X_test, y_test, possible_C, possible_gamma, possible_q):
#     best_params = None
#     best_test_accuracy = 0.0
#     results = pd.DataFrame(columns=['C', 'gamma', 'q', 'train_accuracy', 'test_accuracy', 'kkt', 'time'])
    
#     for C, gamma, q in it.product(possible_C, possible_gamma, possible_q):
#         try:
#             print(C,gamma,q)
#             tol, tol1, tol2 = 1e-3, 1e-3, 1e-3
#             ker='Polynomial'
#             start_time = time.time()
#             alpha, kkt = decomposition_alg(X_train, X_train,y_train,gamma, ker, tol1, tol2, q, C)
#             end_time = time.time()

#             y_train_pred = decision_function(X_train, alpha, X_train, y_train, gamma, C, tol, ker)
#             y_test_pred = decision_function(X_test, alpha, X_train, y_train, gamma, C, tol, ker)

#             train_accuracy = accuracy_score(y_train, y_train_pred)
#             test_accuracy = accuracy_score(y_test, y_test_pred)

#             execution_time = end_time - start_time

#             results.loc[len(results.index)] = [C, gamma, q, train_accuracy, test_accuracy, kkt, execution_time]

#             if test_accuracy > best_test_accuracy:
#                 best_params = (C, gamma, q)
#                 best_test_accuracy = test_accuracy

#         except Exception as e:
#             results.loc[len(results.index)] = {
#                 'C': C, 
#                 'gamma': gamma, 
#                 'q': q, 
#                 'train_accuracy': None, 
#                 'test_accuracy': None,
#                 'kkt': None,  
#                 'time': None,
#                 'error': str(e)  # Registrazione della descrizione dell'errore
#             }

#     return results, best_params

# def grid_search_custom_SVM(X_train, y_train, X_val, y_val, possible_C, possible_gamma, possible_q, K, random_seed):
#     tol, tol1, tol2 = 1e-5, 1e-5, 1e-5
#     best_params = None
#     best_val_accuracy = 0.0

#     # Aggiunta della colonna 'final_kkt' al DataFrame
#     results = pd.DataFrame(columns=['C', 'gamma', 'q', 'train_accuracy', 'val_accuracy', 'time', 'kkt_values', 'final_kkt'])

#     # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=random_seed)

#     for C, gamma, q in it.product(possible_C, possible_gamma, possible_q):
#         try:
#             train_accuracy = 0.0
#             val_accuracy = 0.0
#             total_time = 0.0
#             kkt_values = []

#             for k in range(K):
#                 ker='Polynomial'
#                 start_time = time.time()
#                 alpha, kkt = decomposition_alg(X_train, X_train, y_train, gamma, 'Polynomial', tol1, tol2, q, C)
#                 end_time = time.time()
#                 total_time += end_time - start_time

#                 y_train_pred = decision_function(X_train, alpha, X_train, y_train, gamma,C, tol,ker)
#                 y_val_pred = decision_function(X_val, alpha, X_train, y_train, gamma, C, tol, ker)

#                 train_accuracy += (1.0 / K) * accuracy_score(y_train, y_train_pred)
#                 val_accuracy += (1.0 / K) * accuracy_score(y_val, y_val_pred)
#                 kkt_values.append(kkt)
#                 print(k)
#             # Calcolo del final_kkt come l'ultimo valore KKT della serie
#             final_kkt = kkt_values[-1] if kkt_values else None

#             results.loc[len(results.index)] = [C, gamma, q, train_accuracy, val_accuracy, total_time, kkt_values, final_kkt]
    
#             if val_accuracy > best_val_accuracy:
#                 best_params = (C, gamma, q)
#                 best_val_accuracy = val_accuracy
#         except Exception as e:
            
#             error_data = {
#                 'C': C, 
#                 'gamma': gamma, 
#                 'q': q, 
#                 'train_accuracy': None, 
#                 'val_accuracy': None, 
#                 'time': None, 
#                 'kkt_values': None, 
#                 'final_kkt': None,
#                 'error': 'errore'  # Aggiunta di un campo 'error' per registrare la descrizione dell'errore
#             }
#             results.loc[len(results.index)] = error_data
    
#     return results, best_params

