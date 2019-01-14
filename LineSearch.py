import numpy as np
from MLP import *
from Utility import *
import math

"""
Mette due matrici M, N di dimensione rispettivamente pari a m,n, in un unico vettore di dimensione (m+n,1)
"""
def vectorize(M,N):

    m = M.shape[0] * M.shape[1]
    n = N.shape[0] * N.shape[1]

    v = np.zeros((m + n,1))

    offset = 0
    for (idx_row,row) in enumerate(M):
        for (idx_col,element) in enumerate(row):

            v[offset] = element
            offset += 1

    for (idx_row, row) in enumerate(N):
        for (idx_col, element) in enumerate(row):
            v[offset] = element
            offset += 1

    return v


def f2phi(alpha,mlp,X,T,grad_W_h,grad_W_o):

    #PESI ATTUALI
    W_h_current = mlp.W_h
    W_o_current = mlp.W_o

    #print("W_h",W_h_current)
    #print("W_o",W_o_current)

    #METTO IL GRADIENTE ATTUALE IN UN UNICO VETTORE
    grad_W = vectorize(-grad_W_h, -grad_W_o)

    #SPOSTO I PESI LUNGO DELTA_W ( = - GRADIENTE)
    mlp.W_h = mlp.W_h + (alpha * grad_W_h)
    mlp.W_o = mlp.W_o + (alpha * grad_W_o)

    #CALCOLO E(w + alpha* delta_W)
    mlp.feedforward(addBias(X))
    phi_alfa = compute_Error(T, mlp.Out_o)

    # GRADIENTE CALCOLATO NEL NUOVO PUNTO
    grad_W_o_new, grad_W_h_new = mlp.backpropagation(addBias(X), T)

    #METTO IL NUOVO GRADIENTE SOTTO FORMA DI UN UNICO VETTORE
    grad_W_new = vectorize(-grad_W_h_new, -grad_W_o_new)

    #CALCOLO LA DERIVATA DI PHI
    phi_prime = float(np.dot(grad_W_new.T, -grad_W))

    #RIMETTO I PESI COME ERANO ALL'INIZIO DELLA FUNZIONE
    mlp.W_h = W_h_current
    mlp.W_o = W_o_current

    return phi_alfa,phi_prime

"""
Controlla se la condizione di armijio è soddisfatta
"""
def check_armijio(phi_zero,phi_prime_zero,phi_alpha,m1,alpha):

    assert m1 < 1
    assert m1 > 0

    return phi_alpha <= (phi_zero + m1*alpha*phi_prime_zero)

"""
Controlla se la condizione di strong Wolfe è soddisfatta
"""
def check_strong_wolfe(phi_prime_alpha,phi_prime_zero,m2):

    assert m2 < 1
    assert m2 > 0

    return math.fabs(phi_prime_alpha) <= -m2 * phi_prime_zero


"""
AWLS
 NOTA: ordine dW_h, dW_o come matrici sul foglio...
"""
def AWLS(mlp, X, T, error_MSE, dW_h, dW_o, alpha_0=0.01, max_it=100, m1=0.001, m2=0.75, tau=0.9, eps=1e-6,mina=1e-12):
    phi_0 = error_MSE
    gradE = vectorize( -dW_h, -dW_o)
    phi_p_0 = - np.linalg.norm(gradE)**2
    alpha = alpha_0
    do_quadratic_int = False

    for it in range(max_it):
        phi_alpha, phi_p_alpha = f2phi(alpha, mlp, X, T, dW_h, dW_o) #TODO... modifica func f2phi...
        arm = check_armijio(phi_0, phi_p_0, phi_alpha, m1, alpha)
        s_wolf = check_strong_wolfe(phi_p_alpha, phi_p_0, m2)

        if arm and s_wolf: # love by RS e sopprattutto Michele =) (NON LO STIA A SENTI' PROFFE !!!!!)
            print("Soddisfatta AW")
            break

        if phi_p_alpha >= 0:
            #   chiama Quadratic interpolation!
            print("phi_p_alpha >= 0")
            do_quadratic_int = True
            break

        print("Iterazione %s: Alpha = %s" % (it, alpha))
        alpha = alpha / tau

    if do_quadratic_int:
        alpha = quadratic_interpolation(alpha, mlp, X, T, dW_h, dW_o, gradE, phi_0, phi_p_0, m1, m2, max_it,
                                    eps, mina)

    return alpha


def quadratic_interpolation(alpha_0, mlp, X, T, dW_h, dW_o,gradE,phi_0,phi_p_0,m1=0.001,m2=0.9,max_it=100,epsilon = 1e-6,mina=1e-12):

    alpha_sx = 0
    alpha_dx = alpha_0
    phi_p_sx = phi_p_0
    phi_dx,phi_p_dx = f2phi(alpha_dx, mlp, X, T, dW_h, dW_o)
    norm_gradE = np.linalg.norm(gradE)
    epsilon_prime = epsilon * norm_gradE
    alpha = alpha_0

    print("Faccio interpolazione in [0,%s]"%(alpha_0))
    for it in range(max_it):

        if math.fabs(phi_p_dx) <= epsilon_prime:
            print("Derivata dx circa 0")
            break

        if alpha_dx - alpha_sx <= mina:
            print("Alpha sx e dx troppo vicini vicini")
            break

        alpha = ((alpha_sx * phi_p_dx) - (alpha_dx* phi_p_sx)) / (phi_p_dx  - phi_p_sx)
        print("Iterazione %s) Alpha Interpolazione = %s"%(it,alpha))

        phi_alpha, phi_p_alpha = f2phi(alpha, mlp, X, T, dW_h, dW_o)

        arm = check_armijio(phi_0, phi_p_0, phi_alpha, m1, alpha)
        s_wolf = check_strong_wolfe(phi_p_alpha, phi_p_0, m2)

        if arm and s_wolf:  # love by RS e sopprattutto Michele =) (NON LO STIA A SENTI' PROFFE !!!!!)
            print("Soddisfatto AW")
            break

        if phi_p_alpha < 0:
            alpha_sx = alpha
            phi_p_sx = phi_p_alpha

        else:

            alpha_dx = alpha
            phi_p_dx = phi_p_alpha
            if alpha_dx <= mina:
                print("Alpha dx troppo piccolo")
                break

    return alpha

"""
-----------------
TEST VARI.....
----------------
"""
if __name__ == '__main__':
    M = np.array([
        [1,2,3],
        [4,5,6]
    ])

    N = np.array([
        [7,8],
        [9,10],
        [11,12],
        [13,14]
    ])

    print(vectorize(M,N))
    print(vectorize(M,N).shape)

    n_features = 3
    n_hidden = 2
    n_out = 2
    alpha = 0.2

    X = np.array([
        [1, -2,3],
        [4, -5,6]
    ])

    Y = np.array([
        [0,1],
        [1,0]
    ])

    mlp = MLP(n_features,n_hidden,n_out,TanhActivation(),LinearActivation())
""""
    print("Wh =\n",mlp.W_h)
    print("Wo =\n",mlp.W_o)
    print ("size Wh:", mlp.W_h.shape)
    print ("size Wo:", mlp.W_o.shape)
    mlp.feedforward(addBias(X))
    err = compute_Error(Y,mlp.Out_o)
    print("Errore = ",err)
    grad_W_o,grad_W_h = mlp.backpropagation(addBias(X),Y)
    print("gradWh =\n",grad_W_h)
    print("gradWo =\n",grad_W_o)
    print ("size gradWh:", grad_W_h.shape)
    print ("size gradWo:", grad_W_o.shape)

    phi, phip = f2phi(alpha,mlp,X,Y,grad_W_h,grad_W_o)
    print("phi(alpha) ", phi)
    print("phi'(alpha) ", phip)


    a = check_armijio(0,1,0.1,0.11,0.2)
    print(a)

    a = check_strong_wolfe(0, 1, 0.1)
    print(a)
"""

mlp.feedforward(addBias(X))
dW_o, dW_h  = mlp.backpropagation(addBias(X), Y)
alpha = AWLS(mlp, X, Y, compute_Error(Y,mlp.Out_o), dW_h, dW_o,m1=0.9)
print("Miglior alpha con AWLS= ",alpha)