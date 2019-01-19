from Utilities.UtilityCM import *
from Trainers.Training import *
import numpy as np


class TrainBackPropLS(Training):

    def __init__(self):
        return

    def train(self,mlp,X, T, X_val, T_val, n_epochs = 1000, eps = 1e-6, threshold = 0.5, suppress_print = False):


        assert n_epochs > 0
        assert eps > 0

        epoch = 0
        norm_gradE_0 = 0.
        eps_prime = 0.
        norm_gradE = 0.
        E = 0.
        gradE_h = None
        gradE_o = None
        done_max_epochs = False #Fatte numero massimo iterazioni
        found_optimum = False #Gradiente minore o uguale a eps_prime

        while (not done_max_epochs) or (not found_optimum):

            if epoch == 0:
                E = compute_obj_function(mlp, X, T, mlp.lambd)
                gradE_h, gradE_o = compute_gradient(mlp, X, T, mlp.lambd)

                norm_gradE_0 = np.linalg.norm(gradE_h) ** 2 + np.linalg.norm(gradE_o) ** 2

                eps_prime = eps * norm_gradE_0
                norm_gradE = norm_gradE_0

            else:
                E = compute_obj_function(mlp, X, T, mlp.lambd)
                gradE_h, gradE_o = compute_gradient(mlp, X, T, mlp.lambd)
                norm_gradE = np.linalg.norm(gradE_h) ** 2 + np.linalg.norm(gradE_o) ** 2

            #CONTROLLO GRADIENTE
            if norm_gradE < eps_prime:
                found_optimum = True

            if not found_optimum:
                #LINE_SEARCH


                #AGGIORNAMENTO
                dW_o_new = -mlp.eta * gradE_o + mlp.alfa * mlp.dW_o_old
                mlp.W_o = mlp.W_o + dW_o_new

                dW_h_new = -mlp.eta * gradE_h + mlp.alfa * mlp.dW_h_old
                mlp.W_h = mlp.W_h + dW_h_new

                mlp.dW_o_old = dW_o_new
                mlp.dW_h_old = dW_h_new


                epoch += 1
                #CONTROLLO EPOCHE
                if epoch >= n_epochs:
                    done_max_epochs = True

        if found_optimum:
            vettore_hidden = np.reshape(mlp.W_h,(-1,1))
            vettore_out = np.reshape(mlp.W_o,(-1,1))
            vettore_finale = np.concatenate((vettore_hidden,vettore_out),axis=0)
            print("TROVATO OTTIMO:\nE = %3f\nnorma gradE/gradE_0=%3f, W_star=\n%s"%(E,norm_gradE/norm_gradE_0,vettore_finale.T))

        elif done_max_epochs:

            print("Terminato per numero massimo di iterazioni")
            vettore_hidden = np.reshape(mlp.W_h, (-1, 1))
            vettore_out = np.reshape(mlp.W_o, (-1, 1))
            vettore_finale = np.concatenate((vettore_hidden, vettore_out), axis=0)
            print("VALORI FINALI(NON OTTIMI):\nE = %3f\nnorma gradE/gradE_0 =%3f\nW_star=\n%s" % (
            E, norm_gradE / norm_gradE_0, vettore_finale.T))








