"""
Questo file contiene la classe MLP preposta ad implementare la rete neurale;
- Ogni elemento è Vettoriazzato
- Non necessita di classi come Neuron o Layers
- Usa le classi/file: Utility & ActivationFunction

"""

import numpy as np
from Utility import *
from Activation_Functions import *


class MLP:
    # Costruttore classe con stati
    ## NOTA: Inseriti Pesi con bias
    def __init__(self,n_feature, n_hidden, n_output, activation_h, activation_o, eta=0.1, lambd=0, alfa=0.75):
        # Valori scalari
        #self.n_input = n_input  # righe di X
        self.n_feature = n_feature  # colonne di X, oppure neuroni input
        self.n_hidden = n_hidden
        self.n_output = n_output

        # Vettorizzato: Matrici
        ## NOTA: Indico gli indici delle dimensioni delle matrici/vettori
        self.W_h = init_Weights(n_hidden, n_feature, fan_in=False, range_start=-0.7,
                                range_end=0.7)  # (n_neuroni_h x n_feature +1)
        self.W_o = init_Weights(n_output, n_hidden, fan_in=False, range_start=-0.7,
                                range_end=0.7)  # (n_neuroni_o x n_neuroni_h +1)
        self.Out_h = None  # (n_esempi x n_neuroni_h)
        self.Out_o = None  # (n_esempi x n_neuroni_o) //Per Monk quindi è un vettore
        self.Net_h = None  # (n_esempi x n_neuroni_h)
        self.Net_o = None  # (n_esempi x n_neuroni_o) //Per Monk quindi è un vettore

        # Si specifica il tipo di f. attivazione dei neuroni
        self.activation_h = activation_h
        self.activation_o = activation_o

        # Hyperparameter!
        self.eta = eta  # learning rate
        self.lambd = lambd  # regolarizzazione-penalityTerm
        self.alfa = alfa  # momentum

        # Lista per avere il plot LC & Accuracy
        ## NOTA: Ultimi elementi della lista sono utili per la fase Grid Search 
        self.errors_tr = []
        self.accuracies_tr = []
        self.errors_vl = []
        self.accuracies_vl = []

        # Servono nella fase di train->backperopagation; delta vecchio dei pesi hidden e output
        self._dW_o_old = np.zeros(self.W_o.shape)
        self._dW_h_old = np.zeros(self.W_h.shape)

    # X già con bias
    def feedforward(self, X):
        # Calcolo hidden layer
        self.Net_h = np.dot(X, self.W_h.T)
        self.Out_h = self.activation_h.compute_function(self.Net_h)  # Output_h=f(Net_h)

        # Calcolo output layer
        Out_h_bias = addBias(self.Out_h)
        self.Net_o = np.dot(Out_h_bias, self.W_o.T)
        self.Out_o = self.activation_o.compute_function(
            self.Net_o)  # Output_o=f(Net_o)=>Classificazione rete; Per Monk è vettore

    # X già con bias
    def backpropagation(self, X, T):
        assert T.shape == self.Out_o.shape

        # Calcolo della f'(Net_o), calcolo delta_neuroneOutput, calcolo delta peso
        ## NOTA: vedere slide Backprop.
        grad_f_o = self.activation_o.compute_function_gradient(self.Out_o)
        diff = (T - self.Out_o)
        delta_o = np.multiply(diff, grad_f_o)  # elemento-per-elemento
        Out_h_bias = addBias(self.Out_h)
        delta_W_o = np.dot(delta_o.T, Out_h_bias)

        # Calcolo della f'(Net_h), calcolo delta_o*pesi_interessati, calcolo delta hidden layer
        ## NOTA: vedere slide Backprop.
        grad_f_h = self.activation_h.compute_function_gradient(self.Out_h)
        W_o_nobias = removeBias(self.W_o)
        sp_h = np.dot(delta_o, W_o_nobias)
        delta_h = np.multiply(sp_h, grad_f_h)  # elemento-per-elemento
        delta_W_h = np.dot(delta_h.T, X)

        return delta_W_o /X.shape[0], delta_W_h / X.shape[0]

    # Train usando la Backprop: The Basic Alg. (vedi Slide Corso ML)
    ## NOTA: solo lista error_tr è stata sviluppata per vedere sul Monk2 LearningCurve e se ok!! IL RESTO TODO...
    def train(self, X, T, n_epochs=1000, eps=10 ^ (-3)):
        assert X.shape[0] == T.shape[0]
        # 1) Init pesi e iperparametri // fatto nel costruttore

        # 4) Condizioni di arresto
        error_MSE = 100
        for epoch in range(n_epochs):

            if (error_MSE < eps):
                break
            # 2) Effettuo la feedfoward, calcolo MSE, calcolo delta_W usando backpropagation
            self.feedforward(X)
            error_MSE = compute_Error(T, self.Out_o)
            self.errors_tr.append(error_MSE)
            dW_o, dW_h = self.backpropagation(X, T)

            # 3) Upgrade weights
            dW_o_new = self.eta * dW_o + self.alfa * self._dW_o_old
            self.W_o = self.W_o + dW_o_new - (self.lambd * self.W_o)

            dW_h_new = self.eta * dW_h + self.alfa * self._dW_h_old
            self.W_h = self.W_h + dW_h_new - (self.lambd * self.W_h)

            self.dW_o_old = dW_o_new
            self.dW_h_old = dW_h_new

            print("Epoch %s/%s) TR Error : %s"%(epoch+1,n_epochs,error_MSE))

            # print("aggiornamento W_h", self.dW_h_old)
            # print("W_h new", self.W_h)
            # print("aggiornamento W_o", self.dW_o_old)
            # print("W_o new", self.W_o)

    def predict_class(self, X, treshold=0.5):
        self.feedforward(X)
        predictions = np.zeros(self.Out_o.shape)
        predictions[self.Out_o >= treshold] = 1
        return predictions

    def predict_value(self):
        return
