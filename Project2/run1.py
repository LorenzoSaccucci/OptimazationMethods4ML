from fun1 import *

train = pd.read_csv('fashion-mnist_train.csv')
test = pd.read_csv('fashion-mnist_test.csv')

indexes = [2,3]

train_set = train[train['label'].isin(indexes)]
test_set = test[test['label'].isin(indexes)]

X = train_set.values[:,1:]
Y = train_set.values[:,0]

X_test = test_set.values[:,1:]
Y_test = test_set.values[:,0]

ind_2 = np.where(Y==2)
ind_2_Test = np.where(Y_test==2)

ind_3 = np.where(Y==3)
ind_3_Test = np.where(Y_test==3)

X2 = X[ind_2[0][:1000]]
Y2 = np.ones(1000)

X3 = X[ind_3[0][:1000]]
Y3 = -np.ones(1000)

X2test = X[ind_2_Test[0][:200]]
Y2test = np.ones(X2test.shape[0])

X3test = X[ind_3_Test[0][:200]]
Y3test = -np.ones(X3test.shape[0])

X = np.concatenate((X2,X3))
Y = np.concatenate((Y2,Y3))

X_Test = np.concatenate((X2test,X3test))
Y_Test = np.concatenate((Y2test,Y3test))


# Y = np.vectorize(map_labels)(Y)
Y_test = np.vectorize(map_labels)(Y_test)

scale = MinMaxScaler()
X_train = scale.fit_transform(X)
X_test = scale.transform(X_Test)

print('')
choice = input('Write G to use a Gaussian kernel, P for a Polynomial kernel: ')
while choice.lower() not in ['g', 'p']:
    choice = input('Try again. Write G to use a Gaussian kernel, P for a Polynomial one: ')

if choice.lower() == 'g':
    Ker = 'Gaussian'
    gamma = 0.01
    C = 10
    tol = 1e-3

elif choice.lower() == 'p':
    Ker = 'Polynomial'
    gamma = 2
    C = 10
    tol = 1e-3

print('')
print(f'''
You have chosen a {Ker} kernel, with parameters: 
gamma = {gamma} 
C = {C}
The optimization procedure is now running. 
''')
print('')

# Computing the optimal solution
start = time.time()
alpha_star = alpha_opt(X_train, Y, gamma, C, Ker)
end = time.time()
running_time=end-start

y_pred_train = decision_function(X_train, alpha_star, X_train, Y, gamma, C, tol, Ker)
y_pred_test = decision_function(X_test, alpha_star, X_train, Y, gamma, C, tol, Ker)

print('')
print('Time required for the optimization procedure:', round(running_time, 2), 'seconds')
print('')
print('Accuracy on the training set:',round(accuracy_score(Y, y_pred_train),2)*100, '%')
print('')
print('Accuracy on the test set:', round(accuracy_score(Y_test, y_pred_test)*100, 3), '%')



Q = np.outer(Y, Y) * K(X_train, X_train, gamma, Ker)
P = X_train.shape[0]

tol1 = 1e-2
tol2 = 1e-2

S = S(alpha_star, Y, C, tol1, tol2)
R = R(alpha_star, Y, C, tol1, tol2)


grad = Q @ alpha_star - np.ones(P)

m = max((-grad * Y)[R])
M = min((-grad * Y)[S])
delta = m - M

fo = obj_func(X_train,Y, alpha_star, gamma, Ker)

print('')
print("Final obj:", fo)
print('')
print('KKT violation (m - M) equal to:', delta)
print('')
input('Press ENTER to visualize the confusion matrix for the test set.')

cm = confusion_matrix(Y_test, y_pred_test)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap='viridis', 
            xticklabels=['Predicted -1', 'Predicted 1'], 
            yticklabels=['Actual -1', 'Actual 1'])

plt.ylabel('Actual Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.title('Confusion Matrix', fontsize=15)
plt.show()

# possible_C_values = [ 1, 3, 5, 10, 20, 50, 100, 500, 1000, 5000]

# # Intervalli più densi e estesi per gamma
# possible_gamma_values = [1,2,3,4,5] #[0.0001, 0.001, 0.01, 0.1, 1]

# start = time.time()
# results, best_params = grid_search_custom_SVM_P(X_train, Y, possible_C_values, possible_gamma_values, K=5, random_seed=1893639)
# end = time.time()

# print("Results:")
# print("Best Parameters:", best_params)
# print('Impiega:', end-start) 