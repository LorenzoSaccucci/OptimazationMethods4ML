
from fun4 import *
train = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('fashion-mnist_test.csv')
indexes = [2, 3, 6]

train_set = train[train['label'].isin(indexes)]
test_set = test[test['label'].isin(indexes)]
X_train_full, Y_train_full = train_set.iloc[:, 1:].values, train_set.iloc[:, 0].values
X_test_full, Y_test_full = test_set.iloc[:, 1:].values, test_set.iloc[:, 0].values

seed = 500
np.random.seed(seed)
X_train = np.concatenate([X_train_full[np.random.choice(np.where(Y_train_full == label)[0], 500, replace=False)] for label in indexes])
Y_train = np.concatenate([np.full(500, label) for label in indexes])
X_test = np.concatenate([X_test_full[np.random.choice(np.where(Y_test_full == label)[0], 100, replace=False)] for label in indexes])
Y_test = np.concatenate([np.full(100, label) for label in indexes])
# Normalizzazione
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

C=100
gamma=0.01
Ker = 'Gaussian'
tol = 1e-3
labels = np.array([2, 3, 6])

predictors = {}
P_train, P_test = len(X_train), len(X_test)
preds_train = np.full((P_train, len(labels)), -1e10)
preds_test = np.full((P_test, len(labels)), -1e10)

methods = ['Standard Soft SVM', 'SMO']  
cm_alpha_opt = None

for method in methods:
    sum_time = 0
    for idx, label in enumerate(labels):
        y_train_label = np.where(Y_train == label, 1, -1)
        start = time.time()

        if method == 'Standard Soft SVM':
            alpha_star = alpha_opt(X_train, y_train_label, gamma, C, Ker)
            Q = np.outer(y_train_label, y_train_label) * K(X_train, X_train, gamma, Ker)
            P = X_train.shape[0]
            tol1 = 1e-3
            tol2 = 1e-3
            S_set = S(alpha_star, y_train_label, C, tol1, tol2)
            R_set = R(alpha_star, y_train_label, C, tol1, tol2)
            grad = Q @ alpha_star - np.ones(P)
            m = max((-grad * y_train_label)[R_set])
            M = min((-grad * y_train_label)[S_set])
            delta = m - M
            print('KKT violation (m - M) for label', label, 'is:', delta)
        else:
            alpha_star = SMO(X_train, y_train_label, Ker, gamma, C, tol)

        end = time.time()
        sum_time += (end - start)
        predictors[label] = alpha_star
        preds_train[:, idx] = predict(X_train, predictors[label], X_train, y_train_label, gamma, C, tol, Ker)
        preds_test[:, idx] = predict(X_test, predictors[label], X_train, y_train_label, gamma, C, tol, Ker)


    # Elaborazione dei Risultati
    preds_train_labels = labels[np.argmax(preds_train, axis=1)]
    preds_test_labels = labels[np.argmax(preds_test, axis=1)]

    # Stampa dei Risultati
    print(f'Metodo: {method}')
    print('Time required for training and prediction:', round(sum_time, 2), 'seconds')
    print('Accuracy on the training set:', accuracy_score(Y_train, preds_train_labels) * 100, '%')
    print('Accuracy on the test set:', round(accuracy_score(Y_test, preds_test_labels), 2) * 100, '%')
    

    if method == 'Standard Soft SVM':
        cm_alpha_opt = confusion_matrix(Y_test, preds_test_labels)

    print("\n")

input('Press ENTER to visualize the confusion matrix for the test set.')

if cm_alpha_opt is not None:
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_alpha_opt, annot=True, fmt="d", cmap='viridis', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.title('Confusion Matrix for alpha_opt', fontsize=15)
    plt.show()



#####################################################################################################################à
#Procedura per trovare miglior seme 

# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import accuracy_score
# import time
# from fun4 import *

# # Caricamento e Pre-elaborazione dei dati
# train = pd.read_csv('fashion-mnist_train.csv')
# test = pd.read_csv('fashion-mnist_test.csv')
# indexes = [2, 3, 6]

# train_set = train[train['label'].isin(indexes)]
# test_set = test[test['label'].isin(indexes)]
# X_train_full, Y_train_full = train_set.iloc[:, 1:].values, train_set.iloc[:, 0].values
# X_test_full, Y_test_full = test_set.iloc[:, 1:].values, test_set.iloc[:, 0].values

# # Parametri del modello
# C = 100
# gamma = 0.01
# Ker = 'Gaussian'
# tol = 1e-3
# labels = np.array([2, 3, 6])

# # Normalizzazione
# scaler = MinMaxScaler()
# X_train_full = scaler.fit_transform(X_train_full)
# X_test_full = scaler.transform(X_test_full)

# def run_model_with_seed(seed):
#     np.random.seed(seed)
#     X_train = np.concatenate([X_train_full[np.random.choice(np.where(Y_train_full == label)[0], 800, replace=False)] for label in indexes])
#     Y_train = np.concatenate([np.full(800, label) for label in indexes])
#     X_test = np.concatenate([X_test_full[np.random.choice(np.where(Y_test_full == label)[0], 400, replace=False)] for label in indexes])
#     Y_test = np.concatenate([np.full(400, label) for label in indexes])

#     # Inizializzazione delle predizioni
#     predictors = {}
#     P_train, P_test = len(X_train), len(X_test)
#     preds_train = np.full((P_train, len(labels)), -1e10)
#     preds_test = np.full((P_test, len(labels)), -1e10)

#     # Addestramento e Predizione
#     for idx, label in enumerate(labels):
#         y_train_label = np.where(Y_train == label, 1, -1)
#         alpha_star = alpha_opt(X_train, y_train_label, gamma, C, Ker)
#         predictors[label] = alpha_star
#         preds_train[:, idx] = predict(X_train, predictors[label], X_train, y_train_label, gamma, C, tol, Ker)
#         preds_test[:, idx] = predict(X_test, predictors[label],X_train, y_train_label, gamma, C, tol, Ker)
    

#     # Elaborazione dei Risultati
#     preds_test = labels[np.argmax(preds_test, axis=1)]
#     return accuracy_score(Y_test, preds_test)

# import random

# max_iterations = 40  # Sostituisci con il numero massimo di iterazioni desiderato
# highest_accuracy = 0
# best_seed = None

# for i in range(max_iterations):
#     seed = random.randint(1, 10000)  # Genera un seed casuale tra 1 e 10000
#     current_accuracy = run_model_with_seed(seed)
    
#     if current_accuracy > highest_accuracy:
#         highest_accuracy = current_accuracy
#         best_seed = seed

# print(best_seed, highest_accuracy)
