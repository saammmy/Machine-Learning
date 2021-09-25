import numpy as np
import matplotlib.pyplot as plt
from numpy.core.function_base import linspace
from linear_regression import *
from sklearn.datasets import make_regression

n_samples = 200
X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
y = np.asmatrix(y).T
X = np.asmatrix(X)
Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
alphaa=[0.001,0.01,0.05,0.1,0.5,1,1.38]
size=np.size(alphaa)
epoch=linspace(1,200,200)
for i in range(size):
    print(alphaa[i])
    loss=np.zeros(np.size(epoch))
    for j in range(np.size(epoch)):
        n_samples = 200
        X,y = make_regression(n_samples= n_samples, n_features=4, random_state=1)
        y = np.asmatrix(y).T
        X = np.asmatrix(X)
        Xtrain, Ytrain, Xtest, Ytest = X[::2], y[::2], X[1::2], y[1::2]
        alpha=alphaa[i]
        n_epoch=epoch[j]
        w=train(Xtrain, Ytrain, alpha, int(epoch[j]))
        yhat_test=compute_yhat(Xtest,w)
        loss[j]=compute_L(yhat_test,Ytest)
    plt.plot(epoch,loss, label= 'alpha='+str(alphaa[i]))
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.yscale("log")
plt.show()
