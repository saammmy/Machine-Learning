import math
import numpy as np
from linear_regression import *
from sklearn.datasets import make_regression
# Note: please don't add any new package, you should solve this problem using only the packages above.
#-------------------------------------------------------------------------
'''
    Problem 2: Apply your Linear Regression
    In this problem, use your linear regression method implemented in problem 1 to do the prediction.
    Play with parameters alpha and number of epoch to make sure your test loss is smaller than 1e-2.
    Report your parameter, your train_loss and test_loss 
    Note: please don't use any existing package for linear regression problem, use your own version.
'''
#--------------------------
n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.asmatrix(y).T
X = np.asmatrix(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
#########################################
## INSERT YOUR CODE HERE
alpha=0.01
n_epoch=1000
w=train(Xtrain, Ytrain, alpha, n_epoch)
yhat_train=compute_yhat(Xtrain,w)
L_train=compute_L(yhat_train,Ytrain)
print("\n *************************************************")
print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print(" Loss for Alpha: "+str(alpha)+" and epochs= "+str(n_epoch))
print(" ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
print("     Training Loss = "+str(L_train))

yhat_test=compute_yhat(Xtest,w)
L_test=compute_L(yhat_test,Ytest)
print("     Test Loss = "+str(L_test))
print("\n ************************************************* \n")
#########################################

