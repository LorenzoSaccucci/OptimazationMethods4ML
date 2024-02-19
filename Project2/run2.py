from fun2 import *
from sklearn.metrics import confusion_matrix
import time

train = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('fashion-mnist_test.csv')

indexes = [2,3]

train_set = train[train['label'].isin(indexes)]
test_set = test[test['label'].isin(indexes)]

X = train_set.values[:,1:]
Y = train_set.values[:,0]

X_Test = test_set.values[:,1:]
Y_Test = test_set.values[:,0]

ind_2 = np.where(Y==2)
ind_2_Test = np.where(Y_Test==2)

ind_3 = np.where(Y==3)
ind_3_Test = np.where(Y_Test==3)

X2 = X[ind_2[0][:1000]]
Y2 = np.ones(1000)

X3 = X[ind_3[0][:1000]]
Y3 = -np.ones(1000)

X2test = X_Test[ind_2_Test[0][:200]]
Y2test = np.ones(X2test.shape[0])

X3test = X_Test[ind_3_Test[0][:200]]
Y3test = -np.ones(X3test.shape[0])

X = np.concatenate((X2,X3))
Y = np.concatenate((Y2,Y3))

X_Test = np.concatenate((X2test,X3test))
Y_Test = np.concatenate((Y2test,Y3test))

scale = MinMaxScaler()
X_train = scale.fit_transform(X)
X_test = scale.transform(X_Test)

print('')
choice = input('Write G to use a Gaussian kernel, P for a Polynomial kernel: ')
while choice.lower() not in ['g', 'p']:
    choice = input('Try again. Write G to use a Gaussian kernel, P for a Polynomial one: ')

if choice.lower() == 'g':
    Ker = 'Gaussian'
    gamma = 0.1
    C = 20
    tol1 = 1e-3
    tol2 = 1e-3
    q=60

elif choice.lower() == 'p':
    Ker = 'Polynomial'
    gamma = 3
    C = 3 #5
    tol1 = 1e-3
    tol2 = 1e-3
    q=80


print('')
print(f'''
You have chosen a {Ker} kernel, with parameters:
gamma = {gamma} 
C = {C}
q={q}
The optimization procedure is now running. 
''')
print('')

start = time.time()
alpha_star, final_kkt = decomposition_alg(X_train, X_train, Y, gamma, Ker ,tol1, tol2, q, C)
end = time.time()
running_time=end-start

fo = obj_func(X_train,Y, alpha_star, gamma, Ker)

y_pred_train = decision_function(X_train, alpha_star, X_train, Y, gamma, C, tol1,Ker)
y_pred_test = decision_function(X_test, alpha_star, X_train, Y,gamma, C,tol1, Ker)

print('KKT:', final_kkt)
print("Final obj:", fo)
print('')
print('Time required for the optimization procedure:', round(running_time, 2), 'seconds')
print('')
print('Accuracy on the training set:', accuracy_score(Y, y_pred_train)*100, '%')
print('')
print('Accuracy on the test set:', accuracy_score(Y_Test, y_pred_test)*100, '%')
print('')

input('Press ENTER to visualize the confusion matrix for the test set.')

cm = confusion_matrix(Y_Test, y_pred_test)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='viridis', 
            xticklabels=['Predicted -1', 'Predicted 1'], 
            yticklabels=['Actual -1', 'Actual 1'])

plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=15)
plt.show()


# possible_C = [ 1,2,5, 10, 20, 50 ,80, 100]
# possible_gamma = [1,2,3]
# possible_q = [2, 4, 6, 8,10,20,50,60,80]

# # Esegui la grid search
# results, best_params = grid_search_custom_SVM(X_train, Y, X_test, Y_Test, possible_C, possible_gamma, possible_q)
# print(best_params)
# # Salva i risultati in un file Excel
# results.to_excel("svm_grid_search_results_po13.xlsx")