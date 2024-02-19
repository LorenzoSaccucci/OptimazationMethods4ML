from fun3 import *

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

Y_test = np.vectorize(map_labels)(Y_test)

scale = MinMaxScaler()
X_train = scale.fit_transform(X)
X_test = scale.transform(X_test)

print('')
choice = input('Write G to use a Gaussian kernel, P for a Polynomial kernel: ')
while choice.lower() not in ['g', 'p']:
    choice = input('Try again. Write G to use a Gaussian kernel, P for a Polynomial one: ')

if choice.lower() == 'g':
    Kernel = 'Gaussian'
    gamma = 0.01
    C = 10
    tol = 1e-3

elif choice.lower() == 'p':
    Kernel = 'Polynomial'
    gamma = 2
    C = 10
    tol = 1e-3
    
print('')
print(f'''
You have chosen a {Kernel} kernel, with parameters:
gamma = {gamma} 
C = {C}
The optimization procedure is now running. 
''')
print('')
    
print('')
print('')

start = time.time()
res = SMO(X_train, Y, Kernel, gamma, C, tol = tol)
end = time.time()   
alpha_star = res[0] 
y_pred_train = decision_function(X_train, alpha_star, X_train,Y, gamma, C, tol, Kernel)
y_pred_test = decision_function(X_test, alpha_star, X_train, Y, gamma, C, tol, Kernel)


print('')
print('Time required for the optimization procedure:', round(end-start, 2), 'seconds')
print('')
print('Number of iterations:', res[1])
print('')
print('Initial value of the objective function:', 0)
print('')


Q = np.outer(Y, Y) * K(X_train, X_train, gamma, Kernel)
P = X_train.shape[0]
alpha_star = np.array(alpha_star).flatten()
fun_star = 0.5*np.dot(alpha_star, Q @ alpha_star) - np.dot(np.ones(P), alpha_star)

print('Final value of the objective function:', fun_star)
print('')
print('Accuracy on the training set:', accuracy_score(Y, y_pred_train)*100, '%')
print('')
print('Accuracy on the test set:', round(accuracy_score(Y_test, y_pred_test)*100, 3),'%')
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
