from MLP.MLP import *
from Utilities.Utility import *

"""
PER CM
"""

"""
Mette due matrici M, N di dimensione rispettivamente pari a m,n, in un unico vettore di dimensione (m+n,1)
"""

def compute_obj_function(mlp,X,T,lambd):
    mlp.feedforward(X)
    mse = compute_Error(T,mlp.Out_o)
    norm_w = np.linalg.norm(mlp.W_h)**2 + np.linalg.norm(mlp.W_o)**2
    loss = mse + (0.5*lambd* norm_w)
    return loss

"""
PER CM
"""
def compute_gradient(mlp,X, T,lambd):

    m_grad_mse_o, m_grad_mse_h = mlp.backpropagation(X,T)
    grad_mse_o = - m_grad_mse_o
    grad_mse_h = - m_grad_mse_h
    grad_o = grad_mse_o + (lambd * mlp.W_o)
    grad_h = grad_mse_h + (lambd * mlp.W_h)
    return grad_h, grad_o

"""
Trasforma 2 matrici in un unico vettore [X|Y]
"""
def matrix2vec(X,Y):
    X_vett = np.reshape(X, (-1, 1))
    Y_vett = np.reshape(Y, (-1, 1))
    vect = np.concatenate((X_vett, Y_vett), axis=0)
    return vect
